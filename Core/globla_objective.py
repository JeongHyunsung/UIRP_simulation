# objective.py
from __future__ import annotations
from typing import Tuple
import datetime
from Model.tasks import Task
from Model.providers import Provider

DEFAULT_WEIGHTS: Tuple[float, float, float, float, float] = (
    1.0,    # a1 – total time
    1000.0, # a2 – budget penalty
    10000.0,# a3 – deadline penalty
    -0.1,   # b1 – provider profit (negative for minimisation)
    1.0     # b2 – provider idle penalty
)

def calc_objective(
    task: Task,
    scene_id: int,
    prov: Provider,
    *,
    scene_start: datetime.datetime | None = None,    # ★ 추가 (옵션)
    exclude_global_transfer: bool = False,           # ★ 새로 추가: 글로벌 파일 전송 제외 옵션
    weights: Tuple[float, float, float, float, float] = DEFAULT_WEIGHTS
) -> float:
    """
    scene_start 가 굳이 필요 없으면 무시하고,
    나중에 시간 의존 패널티를 넣고 싶을 때 활용할 수 있습니다.
    
    exclude_global_transfer: True인 경우 글로벌 파일 전송 비용을 제외합니다.
    (GPU 내 다중 씬 할당 시 첫 번째 씬이 아닌 경우 사용)
    """
    a1, a2, a3, b1, b2 = weights
    
    # 글로벌 파일 비용 조정
    global_size = 0.0 if exclude_global_transfer else task.global_file_size
    size = global_size + task.scene_size(scene_id)
    
    rate = min(task.bandwidth, prov.bandwidth)
    T_tx = size / rate
    T_cmp = task.scene_workload / prov.throughput
    T_tot = T_tx + T_cmp
    cost = T_tot * prov.price_per_gpu_hour
    profit_rate = prov.price_per_gpu_hour

    window_h = (task.deadline - task.start_time).total_seconds() / 3600.0
    idle = prov.idle_ratio()

    # scene_start 를 아직 쓰지 않더라도, 인자를 받아 둬야 TypeError 가 안 난다
    return (
        a1 * T_tot +
        a2 * max(0.0, cost - task.budget) +
        a3 * max(0.0, T_tot - window_h) +
        b1 * profit_rate +
        b2 * idle
    )

def calc_objective_grouped(
    task: Task,
    scene_ids: list[int],
    prov: Provider,
    *,
    weights: Tuple[float, float, float, float, float] = DEFAULT_WEIGHTS
) -> float:
    """
    GPU에 할당된 씬 그룹의 총 목적함수 값 계산 (글로벌 파일 최적화 적용)
    """
    if not scene_ids:
        return 0.0
    
    a1, a2, a3, b1, b2 = weights
    
    # 그룹 전체의 전송 및 계산 시간
    total_scene_size = sum(task.scene_size(sid) for sid in scene_ids)
    total_size = task.global_file_size + total_scene_size  # 글로벌 파일 1회만
    
    rate = min(task.bandwidth, prov.bandwidth)
    T_tx = total_size / rate
    T_cmp = sum(task.scene_workload / prov.throughput for _ in scene_ids)
    T_tot = T_tx + T_cmp
    cost = T_tot * prov.price_per_gpu_hour
    profit_rate = prov.price_per_gpu_hour

    window_h = (task.deadline - task.start_time).total_seconds() / 3600.0
    idle = prov.idle_ratio()

    return (
        a1 * T_tot +
        a2 * max(0.0, cost - task.budget) +
        a3 * max(0.0, T_tot - window_h) +
        b1 * profit_rate +
        b2 * idle
    )
