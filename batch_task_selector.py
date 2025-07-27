# ================================================================
# batch_task_selector.py
# 상위 몇 개 task를 모아서 동등하게 처리하는 배치 처리 모듈
# ================================================================
from __future__ import annotations
import datetime
from abc import ABC, abstractmethod
from typing import List, Sequence, Optional, Dict, Set
from itertools import combinations, permutations

from tasks import Task
from providers import Providers

# ----------------------------------------------------------------
# 1) 배치 처리 TaskSelector 인터페이스
# ----------------------------------------------------------------
class TaskSelector(ABC):
    @abstractmethod
    def select(self, now: datetime.datetime, waiting: Sequence[Task]) -> List[Task]: ...

class BatchTaskSelector(TaskSelector):
    """상위 몇 개 task를 모아서 동등하게 처리하는 배치 처리기"""
    
    def __init__(self, batch_size: int = 3, 
                 strategy: str = "oldest_first",
                 enable_concurrent_optimization: bool = True):
        """
        Args:
            batch_size: 한 번에 처리할 task 개수
            strategy: 배치 선택 전략 ("oldest_first", "deadline_first", "budget_first")
            enable_concurrent_optimization: 배치 내 task들을 동시 최적화할지 여부
        """
        self.batch_size = batch_size
        self.strategy = strategy
        self.enable_concurrent_optimization = enable_concurrent_optimization
        self.current_batch: List[Task] = []
        self.batch_start_time: Optional[datetime.datetime] = None
    
    def select(self, now: datetime.datetime, waiting: Sequence[Task]) -> List[Task]:
        """배치 단위로 task 선택"""
        if not waiting:
            return []
        
        # 전략에 따라 정렬
        sorted_tasks = self._sort_by_strategy(list(waiting))
        
        # 배치 크기만큼 선택
        batch = sorted_tasks[:self.batch_size]
        
        if self.enable_concurrent_optimization and len(batch) > 1:
            # 동시 최적화를 위한 배치 정보 저장
            self.current_batch = batch
            self.batch_start_time = now
            
            # 배치 내 task들을 동등하게 처리
            return self._prepare_concurrent_batch(batch, now)
        else:
            # 기존 방식대로 순차 처리
            return batch
    
    def _sort_by_strategy(self, tasks: List[Task]) -> List[Task]:
        """전략에 따른 task 정렬"""
        if self.strategy == "oldest_first":
            # 시작 시간이 빠른 순서
            return sorted(tasks, key=lambda t: t.start_time)
        elif self.strategy == "deadline_first":
            # 데드라인이 급한 순서
            return sorted(tasks, key=lambda t: t.deadline)
        elif self.strategy == "budget_first":
            # 예산이 큰 순서 (높은 우선순위)
            return sorted(tasks, key=lambda t: t.budget, reverse=True)
        else:
            # 기본값: 시작 시간 순서
            return sorted(tasks, key=lambda t: t.start_time)
    
    def _prepare_concurrent_batch(self, batch: List[Task], now: datetime.datetime) -> List[Task]:
        """동시 최적화를 위한 배치 준비"""
        # 모든 task를 동등하게 취급하여 반환
        # 실제 동시 최적화는 ConcurrentBatchOptimizer에서 수행
        return batch

# ----------------------------------------------------------------
# 2) 동시 최적화 배치 처리기
# ----------------------------------------------------------------
class ConcurrentBatchOptimizer:
    """배치 내 여러 task를 동시에 최적화하는 클래스"""
    
    def __init__(self, max_concurrent_tasks: int = 3):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.active_batches: Dict[str, List[Task]] = {}  # batch_id -> tasks
        
    def add_batch(self, batch_id: str, tasks: List[Task]) -> None:
        """새로운 배치를 대기란에 추가"""
        if len(tasks) <= self.max_concurrent_tasks:
            self.active_batches[batch_id] = tasks
            print(f"배치 {batch_id}: {len(tasks)}개 task를 동시 최적화 대기란에 추가")
            for task in tasks:
                print(f"  - Task {task.id}: 씬 {task.scene_number}개, 예산 {task.budget:.1f}$, 데드라인 {task.deadline}")
        else:
            print(f"경고: 배치 크기({len(tasks)})가 최대 동시 처리 수({self.max_concurrent_tasks})를 초과")
    
    def get_batch_for_optimization(self, batch_id: str) -> Optional[List[Task]]:
        """최적화할 배치 반환"""
        return self.active_batches.get(batch_id)
    
    def remove_completed_batch(self, batch_id: str) -> None:
        """완료된 배치 제거"""
        if batch_id in self.active_batches:
            completed_tasks = self.active_batches.pop(batch_id)
            print(f"배치 {batch_id} 완료: {len(completed_tasks)}개 task 처리됨")
    
    def optimize_batch_allocation(self, tasks: List[Task], providers: Providers, 
                                sim_time: datetime.datetime) -> Optional[Dict[str, List[int]]]:
        """배치 내 모든 task에 대한 동시 최적화된 GPU 할당"""
        
        if not tasks:
            return None
        
        print(f"\n=== 배치 동시 최적화 시작 ===")
        print(f"Task 수: {len(tasks)}")
        print(f"사용 가능한 GPU 수: {len(providers)}")
        
        # 전체 GPU 풀에서 각 task에 최적 할당을 찾기
        total_scenes = sum(task.scene_number for task in tasks)
        available_gpus = list(range(len(providers)))
        
        if len(available_gpus) < total_scenes:
            print(f"경고: 필요한 GPU 수({total_scenes})가 사용 가능한 GPU 수({len(available_gpus)})보다 많음")
            return None
        
        best_allocation = None
        best_total_objective = float('inf')
        
        # 각 task에 대해 독립적으로 최적 할당 찾기 (단순화)
        task_allocations = {}
        used_gpus: Set[int] = set()
        
        for task in tasks:
            # 현재 task에 사용 가능한 GPU들 (다른 task에서 사용하지 않은 것들)
            available_for_task = [gpu_id for gpu_id in available_gpus if gpu_id not in used_gpus]
            
            if len(available_for_task) < task.scene_number:
                print(f"Task {task.id}: 사용 가능한 GPU 부족")
                continue
            
            # 간단한 greedy 할당: 씬 크기 순으로 GPU 성능 순으로 매칭
            task_allocation = self._allocate_gpus_for_task(task, available_for_task, providers)
            
            if task_allocation:
                task_allocations[task.id] = task_allocation
                used_gpus.update(task_allocation)
                print(f"Task {task.id} 할당: {task_allocation}")
        
        return task_allocations if task_allocations else None
    
    def _allocate_gpus_for_task(self, task: Task, available_gpus: List[int], 
                              providers: Providers) -> Optional[List[int]]:
        """단일 task에 대한 GPU 할당"""
        if len(available_gpus) < task.scene_number:
            return None
        
        # 씬을 크기 순으로 정렬
        scenes_by_size = []
        for scene_id in range(task.scene_number):
            scene_size = task.scene_size(scene_id)
            scenes_by_size.append((scene_id, scene_size))
        scenes_by_size.sort(key=lambda x: x[1], reverse=True)
        
        # GPU를 성능 순으로 정렬
        gpus_by_performance = []
        for gpu_id in available_gpus:
            prov = providers[gpu_id]
            gpus_by_performance.append((gpu_id, prov.throughput))
        gpus_by_performance.sort(key=lambda x: x[1], reverse=True)
        
        # 선택된 GPU 중에서 최고 성능 순으로 할당
        selected_gpus = [gpu_id for gpu_id, _ in gpus_by_performance[:task.scene_number]]
        
        # 큰 씬 -> 고성능 GPU 매칭
        allocation = [0] * task.scene_number
        for i, (scene_id, _) in enumerate(scenes_by_size):
            allocation[scene_id] = selected_gpus[i]
        
        return allocation

# ----------------------------------------------------------------
# 3) 통합 배치 처리 스케줄러
# ----------------------------------------------------------------
class BatchAwareTaskSelector(TaskSelector):
    """배치 처리와 동시 최적화를 통합한 TaskSelector"""
    
    def __init__(self, batch_size: int = 3, 
                 strategy: str = "oldest_first",
                 wait_for_batch_complete: bool = True):
        """
        Args:
            batch_size: 배치 크기
            strategy: 배치 선택 전략
            wait_for_batch_complete: 배치가 완성될 때까지 기다릴지 여부
        """
        self.batch_selector = BatchTaskSelector(batch_size, strategy, True)
        self.batch_optimizer = ConcurrentBatchOptimizer(batch_size)
        self.wait_for_batch_complete = wait_for_batch_complete
        self.pending_batch: List[Task] = []
        
    def select(self, now: datetime.datetime, waiting: Sequence[Task]) -> List[Task]:
        """배치 인식 task 선택"""
        waiting_list = list(waiting)
        
        if not waiting_list:
            return []
        
        # 대기 중인 task가 배치 크기에 도달했거나, 강제 처리 조건
        if (len(waiting_list) >= self.batch_selector.batch_size or 
            not self.wait_for_batch_complete):
            
            # 배치 선택
            selected_batch = self.batch_selector.select(now, waiting_list)
            
            if len(selected_batch) > 1:
                print(f"\n=== 배치 처리 시작 ===")
                print(f"선택된 Task들: {[task.id for task in selected_batch]}")
                print(f"배치 크기: {len(selected_batch)}")
                print(f"총 씬 수: {sum(task.scene_number for task in selected_batch)}")
                
                # 배치를 동시 최적화 대기란에 추가
                batch_id = f"batch_{now.strftime('%Y%m%d_%H%M%S')}"
                self.batch_optimizer.add_batch(batch_id, selected_batch)
            
            return selected_batch
        else:
            # 배치가 완성되지 않았으면 대기
            print(f"배치 대기 중: {len(waiting_list)}/{self.batch_selector.batch_size}")
            return []
    
    def get_batch_optimizer(self) -> ConcurrentBatchOptimizer:
        """배치 최적화기 반환"""
        return self.batch_optimizer

# ----------------------------------------------------------------
# 4) 사용 예시 및 테스트
# ----------------------------------------------------------------
def example_usage():
    """사용 예시"""
    
    # 1. 기본 배치 처리
    print("=== 기본 배치 TaskSelector ===")
    batch_selector = BatchTaskSelector(batch_size=3, strategy="deadline_first")
    
    # 2. 동시 최적화 포함 배치 처리  
    print("\n=== 동시 최적화 배치 TaskSelector ===")
    concurrent_selector = BatchAwareTaskSelector(
        batch_size=3, 
        strategy="oldest_first",
        wait_for_batch_complete=True
    )
    
    print("배치 처리 모듈이 준비되었습니다!")
    print("사용법:")
    print("1. BatchTaskSelector: 단순 배치 처리")
    print("2. BatchAwareTaskSelector: 동시 최적화 포함 배치 처리")
    print("3. 기존 스케줄러에서 selector 매개변수로 사용 가능")

if __name__ == "__main__":
    example_usage() 