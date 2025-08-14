# Core/Scheduler/combo_generator/cpsat.py
from __future__ import annotations
import math
from typing import List, Tuple
from ortools.sat.python import cp_model

from Core.Scheduler.interface import ComboGenerator
from Core.objective import DEFAULT_WEIGHTS

_SCALE = 1000
_BIG   = 10**9

def _build_common_model(t, ps, now):
    S, P = t.scene_number, len(ps)
    TOT  = [[0.0]*P for _ in range(S)]
    COST = [[0.0]*P for _ in range(S)]
    PROF = [[0.0]*P for _ in range(S)]
    for s in range(S):
        for p in range(P):
            prov = ps[p]
            bw   = min(t.bandwidth, prov.bandwidth)
            thr  = getattr(prov, "throughput", 0.0)
            if bw <= 0 or thr <= 0:
                tx = cmp = float("inf")
            else:
                size = t.global_file_size + t.scene_size(s)
                tx = size / bw / 3600.0
                cmp = t.scene_workload / thr
            TOT[s][p]  = tx + cmp
            COST[s][p] = TOT[s][p] * prov.price_per_gpu_hour
            PROF[s][p] = prov.price_per_gpu_hour

    # 이미 사용한 비용(부분 배정)
    spent = 0.0
    for prov in ps:
        for rec in getattr(prov, "schedule", []):
            if len(rec) == 3:
                tid, st, ft = rec
            else:
                tid, _, st, ft = rec
            if tid == t.id:
                spent += ((ft - st).total_seconds()/3600.0) * prov.price_per_gpu_hour
    remaining_budget = max(0.0, t.budget - spent)

    m = cp_model.CpModel()
    x = [[m.NewBoolVar(f"x{s}_{p}") for p in range(P)] for s in range(S)]
    y = [m.NewBoolVar(f"y{s}") for s in range(S)]  # 이번 스텝 배치 여부

    # 한 씬: 0개(미배치) 또는 1개 provider
    for s in range(S):
        m.Add(sum(x[s][p] for p in range(P)) == y[s])

    # 불가능 금지 + 이미 배정된 씬 고정
    for s in range(S):
        assigned = (t.scene_allocation_data[s][0] is not None)
        fixed_p = t.scene_allocation_data[s][1] if assigned else None
        for p in range(P):
            if not math.isfinite(TOT[s][p]):
                m.Add(x[s][p] == 0)
        if assigned:
            m.Add(y[s] == 1)
            for p in range(P):
                m.Add(x[s][p] == (1 if p == fixed_p else 0))
            for p in range(P):
                TOT[s][p]  = 0.0 if p == fixed_p else float("inf")
                COST[s][p] = 0.0 if p == fixed_p else float("inf")
                PROF[s][p] = 0.0

    # provider 시간/비용/윈도우
    tot_int  = [[int(TOT[s][p]*3600*_SCALE) if math.isfinite(TOT[s][p]) else _BIG for p in range(P)] for s in range(S)]
    cost_int = [[int(COST[s][p]*_SCALE)     if math.isfinite(COST[s][p]) else _BIG for p in range(P)] for s in range(S)]
    prof_int = [[int(PROF[s][p]*_SCALE)     for p in range(P)] for s in range(S)]

    prov_time = [m.NewIntVar(0, _BIG, f"time_p{p}") for p in range(P)]
    for p in range(P):
        m.Add(prov_time[p] == sum(tot_int[s][p] * x[s][p] for s in range(S)))
    makespan = m.NewIntVar(0, _BIG, "makespan")
    m.AddMaxEquality(makespan, prov_time)

    total_cost = m.NewIntVar(0, _BIG, "total_cost")
    m.Add(total_cost == sum(cost_int[s][p] * x[s][p] for s in range(S) for p in range(P)))
    m.Add(total_cost <= int(remaining_budget * _SCALE))
    window_sec = int((t.deadline - now).total_seconds() * _SCALE)
    m.Add(makespan <= window_sec)

    # 가능하면 최소 1개 이상 배치
    unassigned = [s for s in range(S) if t.scene_allocation_data[s][0] is None]
    has_any_finite = any(tot_int[s][p] < _BIG for s in unassigned for p in range(P))
    if has_any_finite:
        m.Add(sum(y[s] for s in unassigned) >= 1)

    return m, x, y, tot_int, cost_int, prof_int

class CPSatComboGenerator(ComboGenerator):
    def best_combo(self, t, ps, now, ev, verbose=False):
        a1, _, _, b1, _ = DEFAULT_WEIGHTS

        # 공통 제약 구성
        m1, x1, y1, tot_int, cost_int, prof_int = _build_common_model(t, ps, now)

        # 1단계: 배치 수 최대화
        m1.Maximize(sum(y1[s] for s in range(t.scene_number)
                        if t.scene_allocation_data[s][0] is None))
        solver1 = cp_model.CpSolver()
        if verbose:
            solver1.parameters.log_search_progress = True
        solver1.parameters.max_time_in_seconds = 10
        status1 = solver1.Solve(m1)
        if status1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        y_opt = int(sum(solver1.Value(y1[s]) for s in range(t.scene_number)
                        if t.scene_allocation_data[s][0] is None))
        if y_opt == 0:
            return None  # 이번 스텝 배치 불가

        # 2단계: 배치 수를 y_opt 이상으로 고정하고, 시간/수익율 최소화
        m2, x2, y2, tot_int2, cost_int2, prof_int2 = _build_common_model(t, ps, now)
        m2.Add(sum(y2[s] for s in range(t.scene_number)
                   if t.scene_allocation_data[s][0] is None) >= y_opt)

        obj_terms = []
        for s in range(t.scene_number):
            for p in range(len(ps)):
                if tot_int2[s][p] >= _BIG:
                    continue
                obj_scene = int(a1*_SCALE) * tot_int2[s][p] + int(b1*_SCALE) * prof_int2[s][p]
                obj_terms.append(obj_scene * x2[s][p])
        m2.Minimize(sum(obj_terms))

        solver2 = cp_model.CpSolver()
        if verbose:
            solver2.parameters.log_search_progress = True
        solver2.parameters.max_time_in_seconds = 10
        status2 = solver2.Solve(m2)
        if status2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None

        # 해 추출
        cmb: List[int] = []
        for s in range(t.scene_number):
            if t.scene_allocation_data[s][0] is not None:
                cmb.append(t.scene_allocation_data[s][1])
                continue
            if solver2.BooleanValue(y2[s]) == 0:
                cmb.append(-1)
                continue
            chosen = None
            for p in range(len(ps)):
                if solver2.BooleanValue(x2[s][p]):
                    chosen = p; break
            cmb.append(chosen if chosen is not None else -1)

        # 메트릭(참고용)
        prov_tot_h = [0.0]*len(ps)
        total_cost_f = 0.0
        for s, p in enumerate(cmb):
            if p == -1:
                continue
            prov_tot_h[p] += (tot_int[s][p] / (3600*_SCALE))
            total_cost_f += (cost_int[s][p] / _SCALE)
        t_tot = max(prov_tot_h) if prov_tot_h else 0.0
        return cmb, t_tot, total_cost_f
