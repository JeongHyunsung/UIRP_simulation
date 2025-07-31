# CLEAR

from __future__ import annotations
import json, datetime, pprint, argparse
from pathlib import Path
from typing import List, Dict, Any
from Model.tasks import Tasks, Task
from Model.providers import Providers
from Core.scheduler import BaselineScheduler, Assignment
from Core.global_objective import calc_objective
from utils.utils import merge_intervals

class Simulator:
    def __init__(self, cfg_path: str):
        cfg = json.loads(Path(cfg_path).read_text())
        self.tasks = Tasks();      self.tasks.initialize_from_data(cfg["tasks"])
        self.providers = Providers(); self.providers.initialize_from_data(cfg["providers"])
        self.results: List[Assignment] = []

    def schedule(self, sch: BaselineScheduler):
        self.results = sch.run(self.tasks, self.providers)

    @staticmethod
    def _missing(task: Task) -> List[int]:
        return [i for i, (st, _) in enumerate(task.scene_allocation_data) if st is None]

    def _idle(self) -> float:
        if not self.results:
            return 1.0
        s = min(r[2] for r in self.results)
        f = max(r[3] for r in self.results)
        horizon = (f - s).total_seconds() / 3600
        prov_busy = {i: 0.0 for i in range(len(self.providers))}
        for i, p in enumerate(self.providers):
            busy = merge_intervals([(st, ft) for _, _, st, ft in p.schedule])  # 1. 사용될 일이 있나? 
            prov_busy[i] = sum((b2 - b1).total_seconds() / 3600 for b1, b2 in busy)
        return max(0.0, 1.0 - sum(prov_busy.values()) / (horizon * len(self.providers)))

    def evaluate(self) -> Dict[str, Any]:
        if not self.results:
            raise RuntimeError("schedule() 먼저 호출 필요")

        tasks_out: Dict[str, Any] = {}
        finished: List[Assignment] = []
        tot_cost = tot_obj = 0.0

        for t in self.tasks:
            rec = [r for r in self.results if r[0] == t.id]
            miss = self._missing(t)
            cost = sum((r[3] - r[2]).total_seconds() / 3600 *
                       self.providers[r[4]].price_per_gpu_hour for r in rec)
            tot_cost += cost

            if miss:
                tasks_out[t.id] = {"completed": False,
                                   "missing": miss,
                                   "cost": cost,
                                   "budget_ok": cost <= t.budget}
                continue

            starts = [r[2] for r in rec]; finishes = [r[3] for r in rec]
            
            # 글로벌 파일 최적화를 위한 GPU별 그룹핑
            gpu_groups: Dict[int, List[Assignment]] = {}
            for r in rec:
                gpu_idx = r[4]  # provider index
                gpu_groups.setdefault(gpu_idx, []).append(r)
            
            # 각 GPU 그룹별로 objective 계산 (글로벌 파일 최적화 적용)
            for gpu_idx, group_assignments in gpu_groups.items():
                # 시작 시간 순으로 정렬 (첫 번째 씬이 글로벌 파일 비용을 부담)
                group_assignments.sort(key=lambda x: x[2])  # start time으로 정렬
                
                for i, r in enumerate(group_assignments):
                    # 첫 번째 씬은 글로벌 파일 포함, 나머지는 제외
                    exclude_global = (i > 0)
                    tot_obj += calc_objective(
                        t, r[1], self.providers[r[4]], 
                        scene_start=r[2],
                        exclude_global_transfer=exclude_global
                    )
            
            tasks_out[t.id] = {
                "completed": True,
                "completion_h": (max(finishes) - min(starts)).total_seconds() / 3600,
                "cost": cost,
                "budget_ok": cost <= t.budget,
                "deadline_ok": max(finishes) <= t.deadline,
            }
            finished += rec

        makespan = ((max(r[3] for r in finished) - min(r[2] for r in finished)).total_seconds() / 3600
                    if finished else 0.0)

        return {"tasks": tasks_out,
                "makespan_h": makespan,
                "overall_idle_ratio": self._idle(),
                "total_cost": tot_cost,
                "objective_sum": tot_obj}

# ---------------- CLI ----------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--config", default="config.json")
    pa.add_argument("--algo",   default="bf", help="bf | greedy | cp")
    pa.add_argument("-v", action="count", default=0, help="-v / -vv for verbose")
    args = pa.parse_args()

    sim = Simulator(args.config)
    sch = BaselineScheduler(algo=args.algo,
                            verbose=args.v >= 1,
                            time_gap=datetime.timedelta(hours=1))
    sim.schedule(sch)
    pprint.pprint(sim.evaluate(), sort_dicts=False)
