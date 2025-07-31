import datetime
from typing import List, Dict
from Core.Scheduler.interface import Dispatcher

Assignment = tuple[str, int, datetime.datetime, datetime.datetime, int]

class SequentialDispatcher(Dispatcher):
    def dispatch(self, t, cmb, now, ps, ev, verbose):
        out: List[Assignment] = []
        groups: Dict[int, List[int]] = {}
        for sid, p in enumerate(cmb):
            groups.setdefault(p, []).append(sid)

        for p, sids in groups.items():
            prov, cur = ps[p], now
            
            # GPU별 그룹 최적화된 스케줄링
            if len(sids) == 1:
                # 단일 씬인 경우 기존 로직 사용
                sid = sids[0]
                dur, _ = ev.time_cost(t, sid, prov)
                st = prov.earliest_available(dur, cur)
                if st is None:
                    continue
                prov.assign(t.id, sid, st, dur)
                t.scene_allocation_data[sid] = (st, p)
                ft = st + datetime.timedelta(hours=dur)
                out.append((t.id, sid, st, ft, p))
                if verbose:
                    print(f"      scene{sid}->{p} {st.strftime('%m-%d %H:%M')} dur={dur:.2f}")
                cur = ft
            else:
                # 다중 씬인 경우 최적화된 로직 사용
                total_time, _ = ev.time_cost_grouped(t, sids, prov)
                
                # 전체 작업을 위한 GPU 예약
                st = prov.earliest_available(total_time, cur)
                if st is None:
                    continue
                
                # 글로벌 파일 전송 시간 계산
                bw = min(t.bandwidth, prov.bandwidth)
                global_tx_time = t.global_file_size / bw / 3600 if bw > 0 else float("inf")
                
                # 모든 씬 파일 전송 시간 계산
                total_scene_size = sum(t.scene_size(sid) for sid in sids)
                scene_tx_time = total_scene_size / bw / 3600 if bw > 0 else float("inf")
                
                # 전송 완료 시점
                transfer_end = st + datetime.timedelta(hours=global_tx_time + scene_tx_time)
                
                # GPU 전체 기간 예약
                prov.assign(t.id, -1, st, total_time)  # -1은 그룹 전체를 나타냄
                
                # 각 씬의 계산 시간
                scene_compute_time = t.scene_workload / prov.throughput if prov.throughput > 0 else float("inf")
                
                # 각 씬을 순차적으로 스케줄링
                current_time = transfer_end
                for i, sid in enumerate(sids):
                    scene_start = current_time
                    scene_end = scene_start + datetime.timedelta(hours=scene_compute_time)
                    
                    t.scene_allocation_data[sid] = (scene_start, p)
                    out.append((t.id, sid, scene_start, scene_end, p))
                    
                    if verbose:
                        print(f"      scene{sid}->{p} {scene_start.strftime('%m-%d %H:%M')} dur={scene_compute_time:.2f} (optimized)")
                    
                    current_time = scene_end
                
                cur = current_time
        
        return out

class CPSatDispatcher(SequentialDispatcher):
    pass
