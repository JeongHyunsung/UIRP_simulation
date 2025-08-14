import os
import sys
import datetime as dt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Core.Scheduler.combo_generator.brute_force import BruteForceGenerator
from Core.Scheduler.combo_generator.hungarian import HungarianComboGenerator
from Core.Scheduler.metric_evaluator.baseline import BaselineEvaluator
from Model.tasks import Task
from Model.providers import Providers


def _sample_case():
    now = dt.datetime(2024, 1, 1, 0, 0, 0)
    task_data = {
        "id": "T1",
        "global_file_size": 0.0,
        "scene_number": 2,
        "scene_file_size": [0.0, 0.0],
        "scene_workload": 10.0,
        "bandwidth": 10.0,
        "budget": 1000.0,
        "deadline": now + dt.timedelta(hours=10),
        "start_time": now,
    }
    task = Task(task_data)
    prov_data = [
        {
            "throughput": 10.0,
            "price": 1.0,
            "bandwidth": 10.0,
            "available_hours": [(now, now + dt.timedelta(hours=5))],
        },
        {
            "throughput": 5.0,
            "price": 1.0,
            "bandwidth": 10.0,
            "available_hours": [(now, now + dt.timedelta(hours=5))],
        },
    ]
    providers = Providers()
    providers.initialize_from_data(prov_data)
    return task, providers, now


def test_hungarian_matches_bruteforce():
    task, providers, now = _sample_case()
    evaluator = BaselineEvaluator()
    bf = BruteForceGenerator()
    hg = HungarianComboGenerator()

    res_bf = bf.best_combo(task, providers, now, evaluator)
    res_hg = hg.best_combo(task, providers, now, evaluator)
    assert res_hg == res_bf
