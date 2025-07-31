import math
import datetime
from typing import Dict, List
from Core.Scheduler.interface import MetricEvaluator
from Core.objective import calc_objective


class BaselineEvaluator(MetricEvaluator):
    def __init__(self):
        self._c: Dict[tuple, float] = {} # cache

    def _t(self, t, s, p): # 씬별 전송시간, 다중 씬 case 수정필요
        k = ("tx", t.id, s, p)
        if k in self._c:
            return self._c[k]
        bw = min(t.bandwidth, p.bandwidth)
        v = float("inf") if bw <= 0 else (t.global_file_size + t.scene_size(s)) / bw / 3600
        self._c[k] = v
        return v

    def _cpt(self, t, p): #
        k = ("cmp", t.id, p)
        if k in self._c:
            return self._c[k]
        v = float("inf") if p.throughput <= 0 else t.scene_workload / p.throughput
        self._c[k] = v
        return v

    def _t_grouped(self, t, scene_ids, p):
        """GPU에 할당된 씬 그룹의 최적화된 전송 시간 계산 (글로벌 파일 1회만)"""
        scene_ids_tuple = tuple(sorted(scene_ids))
        k = ("tx_grouped", t.id, scene_ids_tuple, p)
        if k in self._c:
            return self._c[k]
        
        bw = min(t.bandwidth, p.bandwidth)
        if bw <= 0:
            v = float("inf")
        else:
            # 글로벌 파일 + 모든 씬 파일 합계 (글로벌 파일은 1회만)
            total_scene_size = sum(t.scene_size(sid) for sid in scene_ids)
            total_size = t.global_file_size + total_scene_size
            v = total_size / bw / 3600
        
        self._c[k] = v
        return v

    def _cpt_grouped(self, t, scene_ids, p):
        """GPU에 할당된 씬 그룹의 총 계산 시간"""
        k = ("cmp_grouped", t.id, tuple(sorted(scene_ids)), p)
        if k in self._c:
            return self._c[k]
        
        if p.throughput <= 0:
            v = float("inf")
        else:
            # 각 씬의 계산 시간 합계
            v = sum(t.scene_workload / p.throughput for _ in scene_ids)
        
        self._c[k] = v
        return v

    def time_cost_grouped(self, t, scene_ids, p):
        """GPU에 할당된 씬 그룹의 총 시간과 비용 (최적화된 계산)"""
        if not scene_ids:
            return 0.0, 0.0
        
        tx_time = self._t_grouped(t, scene_ids, p)
        cmp_time = self._cpt_grouped(t, scene_ids, p)
        total_time = tx_time + cmp_time
        cost = total_time * p.price_per_gpu_hour
        
        return total_time, cost

    def time_cost(self, t, s, p):
        """기존 단일 씬 계산 (하위 호환성 유지)"""
        d = self._t(t, s, p) + self._cpt(t, p)
        return d, d * p.price_per_gpu_hour

    def feasible(self, t, cmb, now, ps):
        grouped: Dict[int, List[int]] = {}
        for sid, p in enumerate(cmb):
            grouped.setdefault(p, []).append(sid)
        
        t_tot, cost = 0.0, 0.0
        for p, sids in grouped.items():
            # 그룹별 최적화된 계산 사용
            group_time, group_cost = self.time_cost_grouped(t, sids, ps[p])
            
            if math.isinf(group_time):
                return False, math.inf, math.inf
            
            t_tot = max(t_tot, group_time)
            cost += group_cost
        
        if cost > t.budget:
            return False, t_tot, cost
        if now + datetime.timedelta(hours=t_tot) > t.deadline:
            return False, t_tot, cost
        return True, t_tot, cost

    def efficiency(self, t, cmb: List[int], ps) -> float:
        total = 0.0
        for sid, pid in enumerate(cmb):
            prov = ps[pid]
            score = calc_objective(t, sid, prov)
            if score == float("inf") or score != score:  # NaN 방지
                return float("-inf")
            total += score
        return -total