# Core/Scheduler/scheduler.py
import datetime
import math
from typing import List
from Model.tasks import Tasks, Task
from Model.providers import Providers
from Core.Scheduler.interface import TaskSelector, MetricEvaluator
from Core.Scheduler.registry import COMBO_REG, DISP_REG

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    def tqdm(iterable=None, **kwargs):
        return iterable

Assignment = tuple[str, int, datetime.datetime, datetime.datetime, int]

class BaselineScheduler:
    def __init__(self, *, algo="bf", time_gap=datetime.timedelta(hours=1),
                 selector: TaskSelector = None,
                 evaluator: MetricEvaluator = None,
                 verbose=False):
        from Core.Scheduler.task_selector.fifo import FIFOTaskSelector
        from Core.Scheduler.metric_evaluator.baseline import BaselineEvaluator
        self.selector = selector or FIFOTaskSelector()
        self.generator = COMBO_REG[algo]()
        self.dispatcher = DISP_REG[algo]()
        self.evaluator = evaluator or BaselineEvaluator()
        self.time_gap = time_gap
        self.verbose = verbose
        self.waiting_tasks: List[Task] = []
        self.results: List[Assignment] = []

    def _feed(self, now, tasks):
        ids = {t.id for t in self.waiting_tasks}
        for t in tasks:
            # 아직 시작시각 도달 & 미완료면 큐에 투입
            if t.start_time <= now and t.id not in ids and any(st is None for st, _ in t.scene_allocation_data):
                self.waiting_tasks.append(t)
        self.waiting_tasks.sort(key=lambda t: t.start_time)

    def _schedule_once(self, now, ps):
        new: List[Assignment] = []
        remain = []
        for t in self.selector.select(now, self.waiting_tasks):
            # [CHANGED] 전부 완료면 건너뜀
            if all(st is not None for st, _ in t.scene_allocation_data):
                continue

            best = self.generator.best_combo(t, ps, now, self.evaluator, verbose=self.verbose)
            if best is None:
                remain.append(t)
                continue
            cmb, t_tot, cost = best
            if self.verbose:
                print(f"[{t.id}] choose {cmb} t={t_tot:.2f}h cost={cost:.1f}$")
            before_missing = sum(1 for st, _ in t.scene_allocation_data if st is None)
            new_assgn = self.dispatcher.dispatch(t, cmb, now, ps, self.evaluator, self.verbose)
            new += new_assgn
            after_missing = sum(1 for st, _ in t.scene_allocation_data if st is None)
            # [CHANGED] 여전히 남은 씬이 있으면 다음 스텝에서도 고려
            if after_missing > 0:
                remain.append(t)
        self.waiting_tasks = remain
        return new

    def run(self, tasks: Tasks, ps: Providers,
            time_start: datetime.datetime | None = None,
            time_end: datetime.datetime | None = None) -> List[Assignment]:
        if time_start is None:
            # provider available_hours 가 비어있을 수 있음에 주의
            starts = []
            for p in ps:
                if getattr(p, 'available_hours', None):
                    starts.append(min(a[0] for a in p.available_hours))
            time_start = min(starts) if starts else min(t.start_time for t in tasks)
        now = time_start
        if time_end is None:
            time_end = max(t.deadline for t in tasks) + datetime.timedelta(days=1)
        steps = math.ceil((time_end - time_start) / self.time_gap)
        pbar = tqdm(range(steps), disable=not self.verbose)
        for _ in pbar:
            self._feed(now, tasks)
            self.results += self._schedule_once(now, ps)
            if all(all(st is not None for st, _ in t.scene_allocation_data) for t in tasks):
                break
            now += self.time_gap
        return self.results
