# Core/Scheduler/combo_generator/hungarian.py
"""Hungarian algorithm based combo generator.

Constructs a cost matrix between scenes and providers and solves
assignment with the Hungarian algorithm. Impossible matches get a
very large cost so they are avoided. Time complexity of the solver is
O(n^3) where n = max(number of scenes, providers).

Compared to brute-force search which explores exponential number of
combinations, this method yields the optimal assignment for the given
pairwise costs while remaining polynomial time. Quality is identical
whenever the overall objective is the sum of individual costs.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from Core.Scheduler.interface import ComboGenerator

_BIG = 10 ** 9
_SKIP_COST = 10 ** 6


def _hungarian(cost: List[List[float]]) -> List[int]:
    """Return optimal column index for each row using Hungarian algorithm."""
    if not cost:
        return []
    n, m = len(cost), len(cost[0])
    transpose = False
    if n > m:
        cost = [list(row) for row in zip(*cost)]
        n, m = m, n
        transpose = True
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    ans = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            ans[p[j] - 1] = j - 1
    if transpose:
        return ans
    return ans


class HungarianComboGenerator(ComboGenerator):
    def best_combo(
        self,
        task,
        providers,
        sim_time,
        evaluator,
        verbose: bool = False,
    ) -> Optional[Tuple[List[int], float, float]]:
        # gather unassigned scenes
        scene_ids = [
            i for i, (st, _) in enumerate(task.scene_allocation_data) if st is None
        ]
        if not scene_ids:
            return None
        S = len(scene_ids)
        P = len(providers)

        cost_mat: List[List[float]] = []
        for idx, sid in enumerate(scene_ids):
            row: List[float] = []
            for prov in providers:
                d, _ = evaluator.time_cost(task, sid, prov)
                if not math.isfinite(d) or d <= 0:
                    row.append(_BIG)
                else:
                    row.append(d)
            # skip columns so that each scene may remain unassigned
            skip_cols = [_BIG] * S
            skip_cols[idx] = _SKIP_COST
            row.extend(skip_cols)
            cost_mat.append(row)

        assign = _hungarian(cost_mat)

        combo = [-1] * task.scene_number
        for sid, col in zip(scene_ids, assign[:S]):
            if col < P:
                combo[sid] = col
            else:
                combo[sid] = -1

        ok, t_tot, cost, deferred, overB, overDL = evaluator.feasible(
            task, combo, sim_time, providers
        )
        if not ok:
            return None
        # efficiency not needed for single assignment; still compute to mimic BF
        evaluator.efficiency(task, combo, providers, sim_time, t_tot, cost, deferred, overB, overDL)
        return combo, t_tot, cost
