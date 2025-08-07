#CLEAR

from __future__ import annotations
import datetime
from typing import Dict, Any, List, Tuple, Optional
from utils.utils import merge_intervals

class Provider:
    def __init__(self, d: Dict[str, Any]):
        self.throughput: float = float(d.get("throughput", 1.0))          # GFLOP/s
        self.price_per_gpu_hour: float = float(d.get("price", 0.0))       # $ 테스트pr
        self.bandwidth: float = float(d.get("bandwidth", 0.0))            # MB/s

        raw = d.get("available_hours", [])
        self.available_hours: List[Tuple[datetime.datetime, datetime.datetime]] = [
            (datetime.datetime.fromisoformat(s) if isinstance(s, str) else s,
             datetime.datetime.fromisoformat(e) if isinstance(e, str) else e)
            for s, e in raw
        ]

        # (task_id, scene_id, start, finish)
        self.schedule: List[Tuple[str, int, datetime.datetime, datetime.datetime]] = []
        
        # 글로벌 파일을 이미 받은 task들 추적
        self.received_global_files: set = set()

    # ---------------------------------------------------
    # 메트릭
    # ---------------------------------------------------
    def idle_ratio(self, start: Optional[datetime.datetime] = None,
                   end: Optional[datetime.datetime] = None) -> float:
        if not self.schedule: return 1.0
        if start is None: start = min(s for *_ ,s,_ in self.schedule)
        if end   is None: end   = max(f for *_,f in self.schedule)
        horizon = (end - start).total_seconds() / 3600.0
        if horizon <= 0: return 1.0
        busy = merge_intervals([(s,f) for *_ ,s,f in self.schedule])
        busy_h = sum((f-s).total_seconds()/3600.0 for s,f in busy)
        return max(0.0, 1.0 - busy_h/horizon)

    # ---------------------------------------------------
    # 스케줄링 헬퍼
    # ---------------------------------------------------
    def earliest_available(self, dur_h: float, after: datetime.datetime) -> Optional[datetime.datetime]:
        """지정 길이(dur_h)를 연속으로 배정할 수 있는 가장 이른 시각(>=after)"""
        for a_s, a_e in sorted(self.available_hours, key=lambda t: t[0]):
            if a_e <= after: continue
            cur = max(a_s, after)

            # 해당 윈도우 내 기존 schedule 과 겹치지 않는 구간 찾기
            clashes = sorted([(s,f) for *_,s,f in self.schedule if s < a_e and f > cur], key=lambda iv: iv[0])
            for s,f in clashes:
                if (s - cur).total_seconds()/3600.0 >= dur_h: return cur
                cur = max(cur, f)
            if (a_e - cur).total_seconds()/3600.0 >= dur_h: return cur
        return None

    def assign(self, task_id: str, scene_id: int,
               start: datetime.datetime, dur_h: float):
        finish = start + datetime.timedelta(hours=dur_h)
        self.schedule.append((task_id, scene_id, start, finish))
        
        # 해당 task의 글로벌 파일을 받았다고 표시
        self.received_global_files.add(task_id)

        # available_hours 업데이트(틈새 제거)
        new=[]
        for s,e in self.available_hours:
            if finish <= s or start >= e: new.append((s,e)); continue
            if s < start: new.append((s,start))
            if finish < e: new.append((finish,e))
        self.available_hours=new

class Providers:
    def __init__(self): self._list: List[Provider]=[]
    def initialize_from_data(self, data):
        self._list=[Provider(d) for d in data]
    def __iter__(self): return iter(self._list)
    def __getitem__(self,i): return self._list[i]
    def __len__(self): return len(self._list)
