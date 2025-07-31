
# baseline_scheduler/scheduler.py
import datetime
from typing import List
from Model.tasks import Tasks, Task
from Model.providers import Providers
from Core.Scheduler.interface import TaskSelector, MetricEvaluator
from Core.Scheduler.registry import COMBO_REG, DISP_REG, METRIC_REG, SELECTOR_REG

Assignment = tuple[str, int, datetime.datetime, datetime.datetime, int]

class BaselineScheduler:
    def __init__(self, *, algo="bf", time_gap=datetime.timedelta(hours=1),
                 selector: TaskSelector = None,
                 evaluator: MetricEvaluator = None,
                 verbose=False):
        self.selector = selector or SELECTOR_REG[algo]()
        self.generator = COMBO_REG[algo]()
        self.dispatcher = DISP_REG[algo]()
        self.evaluator = evaluator or METRIC_REG[algo]()
        self.time_gap = time_gap
        self.verbose = verbose
        self.waiting_tasks: List[Task] = []
        self.results: List[Assignment] = []

    def _feed(self, now, tasks):
        ids = {t.id for t in self.waiting_tasks}
        for t in tasks:
            if t.start_time <= now and t.id not in ids and any(st is None for st, _ in t.scene_allocation_data):
                self.waiting_tasks.append(t)
        self.waiting_tasks.sort(key=lambda t: t.start_time)

    def _schedule_once(self, now, ps):
        new: List[Assignment] = []
        remain = []
        for t in self.selector.select(now, self.waiting_tasks):
            if not all(st is None for st, _ in t.scene_allocation_data):
                remain.append(t)
                continue
            best = self.generator.best_combo(t, ps, now, self.evaluator, verbose=self.verbose)
            if best is None:
                remain.append(t)
                continue
            cmb, t_tot, cost = best
            if self.verbose:
                print(f"[{t.id}] choose {cmb} t={t_tot:.2f}h cost={cost:.1f}$")
            new += self.dispatcher.dispatch(t, cmb, now, ps, self.evaluator, self.verbose)
        self.waiting_tasks = remain
        return new

    def run(self, tasks: Tasks, ps: Providers,
            time_start: datetime.datetime | None = None,
            time_end: datetime.datetime | None = None) -> List[Assignment]:
        if time_start is None:
            time_start = min(min(p.available_hours)[0] for p in ps)
        now = time_start
        if time_end is None:
            time_end = max(t.deadline for t in tasks) + datetime.timedelta(days=1)
        while now < time_end:
            self._feed(now, tasks)
            self.results += self._schedule_once(now, ps)
            if all(all(st is not None for st, _ in t.scene_allocation_data) for t in tasks):
                break
            now += self.time_gap
        return self.results