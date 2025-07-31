from Core.Scheduler.interface import ComboGenerator

class GreedyComboGenerator(ComboGenerator):
    def __init__(self):
        pass

    def _generate_smart_combos(self, t, ps, verbose=False):
        """중복조합으로 GPU 사용 개수 조합 생성 후 greedy 배정"""
        import itertools
        
        # 씬들을 크기 내림차순으로 정렬
        scene_indices_by_size = sorted(
            range(t.scene_number),
            key=lambda i: t.scene_file_sizes[i],
            reverse=True
        )
        
        # GPU들을 성능 내림차순으로 정렬
        gpu_indices_by_performance = sorted(
            range(len(ps)),
            key=lambda i: ps[i].throughput,
            reverse=True
        )
        
        # 중복조합: GPU 사용 개수 조합들 생성
        gpu_count_combos = list(itertools.combinations_with_replacement(
            range(len(ps)), t.scene_number
        ))
        
        # 각 GPU 사용 개수 조합을 실제 씬 배정으로 변환
        combos = []
        for gpu_count_combo in gpu_count_combos:
            combo = [0] * t.scene_number
            
            # GPU 사용 개수 세기
            gpu_counts = [0] * len(ps)
            for gpu_idx in gpu_count_combo:
                gpu_counts[gpu_idx] += 1
            
            # 큰 씬부터 좋은 GPU에 배정 (greedy 원칙)
            assigned_scenes = 0
            for gpu_idx in gpu_indices_by_performance:
                scenes_to_assign = gpu_counts[gpu_idx]
                for _ in range(scenes_to_assign):
                    if assigned_scenes < t.scene_number:
                        scene_idx = scene_indices_by_size[assigned_scenes]
                        combo[scene_idx] = gpu_idx
                        assigned_scenes += 1
            
            combos.append(combo)
        
        if verbose:
            print(f"Generated {len(combos)} combos from {len(gpu_count_combos)} GPU count combinations")
        
        return combos

    def best_combo(self, t, ps, now, ev, verbose=False):
        """
        여러 스마트한 조합을 시도해서 최적 조합 선택
        """
        smart_combos = self._generate_smart_combos(t, ps, verbose)
        
        best = (-1.0, None)
        
        for combo in smart_combos:
            ok, t_tot, cost = ev.feasible(t, combo, now, ps)
            if verbose:
                print(f"    → {combo} ok={ok} t={t_tot:.2f}h cost={cost:.1f}$")
            
            if not ok:
                continue
                
            eff = ev.efficiency(t, t_tot, cost)
            if eff > best[0]:
                best = (eff, (combo, t_tot, cost))
        
        if verbose and best[1]:
            combo, t_tot, cost = best[1]
            print(f"    Best greedy combo: {combo} (eff={best[0]:.3f})")
            
        return None if best[1] is None else best[1] 