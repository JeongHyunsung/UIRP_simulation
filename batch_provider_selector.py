# ================================================================
# batch_provider_selector.py
# 상위 몇 개 GPU/Provider를 모아서 동등하게 처리하는 배치 처리 모듈
# ================================================================
from __future__ import annotations
import datetime
from abc import ABC, abstractmethod
from typing import List, Dict, Set, Optional, Tuple
from itertools import combinations, permutations
import random

from providers import Providers, Provider
from tasks import Task

# ----------------------------------------------------------------
# 1) 배치 처리 ProviderSelector 인터페이스
# ----------------------------------------------------------------
class ProviderSelector(ABC):
    @abstractmethod
    def select_providers(
        self, 
        providers: Providers, 
        sim_time: datetime.datetime,
        required_count: int = None
    ) -> List[int]: ...

class BatchProviderSelector(ProviderSelector):
    """상위 몇 개 GPU/Provider를 모아서 동등하게 처리하는 배치 처리기"""
    
    def __init__(self, 
                 batch_size: int = 5,
                 strategy: str = "performance_first",
                 enable_concurrent_optimization: bool = True,
                 load_balancing: bool = True):
        """
        Args:
            batch_size: 한 번에 고려할 GPU 개수
            strategy: 배치 선택 전략 
                     ("performance_first", "cost_first", "idle_first", "random", "availability_first")
            enable_concurrent_optimization: 배치 내 GPU들을 동시 최적화할지 여부
            load_balancing: 부하 분산을 고려할지 여부
        """
        self.batch_size = batch_size
        self.strategy = strategy
        self.enable_concurrent_optimization = enable_concurrent_optimization
        self.load_balancing = load_balancing
        self.current_batch: List[int] = []
        self.batch_start_time: Optional[datetime.datetime] = None
        self.usage_history: Dict[int, int] = {}  # gpu_id -> usage_count
    
    def select_providers(self, 
                        providers: Providers, 
                        sim_time: datetime.datetime,
                        required_count: int = None) -> List[int]:
        """배치 단위로 Provider 선택"""
        
        if not providers:
            return []
        
        # 사용 가능한 Provider 필터링
        available_providers = self._filter_available_providers(providers, sim_time)
        
        if not available_providers:
            return []
        
        # 전략에 따라 정렬
        sorted_providers = self._sort_by_strategy(available_providers, providers)
        
        # 배치 크기 결정
        effective_batch_size = min(
            self.batch_size,
            len(sorted_providers),
            required_count if required_count else len(sorted_providers)
        )
        
        # 배치 선택
        if self.load_balancing:
            selected_batch = self._select_with_load_balancing(
                sorted_providers, effective_batch_size
            )
        else:
            selected_batch = sorted_providers[:effective_batch_size]
        
        if self.enable_concurrent_optimization and len(selected_batch) > 1:
            # 동시 최적화를 위한 배치 정보 저장
            self.current_batch = selected_batch
            self.batch_start_time = sim_time
            
            # 배치 내 Provider들을 동등하게 처리
            return self._prepare_concurrent_batch(selected_batch, sim_time, providers)
        else:
            # 기존 방식대로 순차 처리
            return selected_batch
    
    def _filter_available_providers(self, providers: Providers, sim_time: datetime.datetime) -> List[int]:
        """사용 가능한 Provider 필터링"""
        available = []
        for i, prov in enumerate(providers):
            try:
                # is_available_at 메서드가 있다면 체크
                if hasattr(prov, 'is_available_at'):
                    if prov.is_available_at(sim_time):
                        available.append(i)
                else:
                    # 메서드가 없으면 기본 체크 (throughput > 0 등)
                    if prov.throughput > 0:
                        available.append(i)
            except:
                # 오류 발생시 사용 가능하다고 가정
                available.append(i)
        
        return available
    
    def _sort_by_strategy(self, provider_ids: List[int], providers: Providers) -> List[int]:
        """전략에 따른 Provider 정렬"""
        
        if self.strategy == "performance_first":
            # 성능(throughput) 높은 순서
            return sorted(provider_ids, 
                         key=lambda i: providers[i].throughput, 
                         reverse=True)
        
        elif self.strategy == "cost_first":
            # 비용(price_per_gpu_hour) 낮은 순서
            return sorted(provider_ids, 
                         key=lambda i: providers[i].price_per_gpu_hour)
        
        elif self.strategy == "idle_first":
            # 유휴율(idle_ratio) 높은 순서 (덜 바쁜 GPU 우선)
            return sorted(provider_ids, 
                         key=lambda i: providers[i].idle_ratio() if hasattr(providers[i], 'idle_ratio') else 0.5, 
                         reverse=True)
        
        elif self.strategy == "availability_first":
            # 대역폭(bandwidth) 높은 순서
            return sorted(provider_ids, 
                         key=lambda i: providers[i].bandwidth, 
                         reverse=True)
        
        elif self.strategy == "random":
            # 랜덤 선택
            shuffled = provider_ids.copy()
            random.shuffle(shuffled)
            return shuffled
        
        else:
            # 기본값: 성능 순서
            return sorted(provider_ids, 
                         key=lambda i: providers[i].throughput, 
                         reverse=True)
    
    def _select_with_load_balancing(self, sorted_providers: List[int], batch_size: int) -> List[int]:
        """부하 분산을 고려한 Provider 선택"""
        
        # 사용 빈도가 낮은 GPU 우선 선택
        providers_with_usage = []
        for provider_id in sorted_providers:
            usage_count = self.usage_history.get(provider_id, 0)
            providers_with_usage.append((provider_id, usage_count))
        
        # 사용 빈도 낮은 순, 동일하면 기존 정렬 순서 유지
        providers_with_usage.sort(key=lambda x: (x[1], sorted_providers.index(x[0])))
        
        selected = [provider_id for provider_id, _ in providers_with_usage[:batch_size]]
        
        # 사용 기록 업데이트
        for provider_id in selected:
            self.usage_history[provider_id] = self.usage_history.get(provider_id, 0) + 1
        
        return selected
    
    def _prepare_concurrent_batch(self, batch: List[int], sim_time: datetime.datetime, 
                                providers: Providers) -> List[int]:
        """동시 최적화를 위한 배치 준비"""
        # 모든 Provider를 동등하게 취급하여 반환
        # 실제 동시 최적화는 ConcurrentProviderOptimizer에서 수행
        return batch

# ----------------------------------------------------------------
# 2) 동시 최적화 Provider 배치 처리기
# ----------------------------------------------------------------
class ConcurrentProviderOptimizer:
    """배치 내 여러 Provider를 동시에 최적화하는 클래스"""
    
    def __init__(self, max_concurrent_providers: int = 5):
        self.max_concurrent_providers = max_concurrent_providers
        self.active_provider_batches: Dict[str, List[int]] = {}  # batch_id -> provider_ids
        self.provider_workload: Dict[int, float] = {}  # provider_id -> current_workload
        
    def add_provider_batch(self, batch_id: str, provider_ids: List[int], 
                          providers: Providers) -> None:
        """새로운 Provider 배치를 대기란에 추가"""
        if len(provider_ids) <= self.max_concurrent_providers:
            self.active_provider_batches[batch_id] = provider_ids
            print(f"Provider 배치 {batch_id}: {len(provider_ids)}개 GPU를 동시 최적화 대기란에 추가")
            for provider_id in provider_ids:
                prov = providers[provider_id]
                print(f"  - GPU {provider_id}: 성능 {prov.throughput:.2f}, 비용 {prov.price_per_gpu_hour:.2f}$/h")
        else:
            print(f"경고: Provider 배치 크기({len(provider_ids)})가 최대 동시 처리 수({self.max_concurrent_providers})를 초과")
    
    def get_provider_batch_for_optimization(self, batch_id: str) -> Optional[List[int]]:
        """최적화할 Provider 배치 반환"""
        return self.active_provider_batches.get(batch_id)
    
    def remove_completed_provider_batch(self, batch_id: str) -> None:
        """완료된 Provider 배치 제거"""
        if batch_id in self.active_provider_batches:
            completed_providers = self.active_provider_batches.pop(batch_id)
            print(f"Provider 배치 {batch_id} 완료: {len(completed_providers)}개 GPU 처리됨")
    
    def optimize_provider_allocation(self, tasks: List[Task], provider_batch: List[int], 
                                   providers: Providers, sim_time: datetime.datetime) -> Optional[Dict[str, List[int]]]:
        """Provider 배치 내에서 task들에 대한 동시 최적화된 할당"""
        
        if not tasks or not provider_batch:
            return None
        
        print(f"\n=== Provider 배치 동시 최적화 시작 ===")
        print(f"Task 수: {len(tasks)}")
        print(f"배치 내 GPU 수: {len(provider_batch)}")
        
        # 현재 Provider 배치의 워크로드 계산
        batch_workload = {}
        for provider_id in provider_batch:
            current_load = self.provider_workload.get(provider_id, 0.0)
            batch_workload[provider_id] = current_load
        
        # 각 task에 대해 최적 Provider 할당
        task_allocations = {}
        
        for task in tasks:
            allocation = self._allocate_providers_for_task(
                task, provider_batch, providers, batch_workload
            )
            
            if allocation:
                task_allocations[task.id] = allocation
                
                # 워크로드 업데이트
                for scene_id, provider_id in enumerate(allocation):
                    if provider_id in batch_workload:
                        # 간단한 워크로드 추정
                        estimated_time = task.scene_workload / providers[provider_id].throughput
                        batch_workload[provider_id] += estimated_time
                
                print(f"Task {task.id} Provider 할당: {allocation}")
        
        # 전역 워크로드 상태 업데이트
        self.provider_workload.update(batch_workload)
        
        return task_allocations if task_allocations else None
    
    def _allocate_providers_for_task(self, task: Task, provider_batch: List[int], 
                                   providers: Providers, current_workload: Dict[int, float]) -> Optional[List[int]]:
        """단일 task에 대한 Provider 배치 내 할당"""
        
        if len(provider_batch) < task.scene_number:
            return None
        
        # 씬을 크기 순으로 정렬
        scenes_by_size = []
        for scene_id in range(task.scene_number):
            scene_size = task.scene_size(scene_id)
            scenes_by_size.append((scene_id, scene_size))
        scenes_by_size.sort(key=lambda x: x[1], reverse=True)
        
        # Provider를 현재 워크로드와 성능을 고려하여 정렬
        providers_by_efficiency = []
        for provider_id in provider_batch:
            prov = providers[provider_id]
            current_load = current_workload.get(provider_id, 0.0)
            
            # 효율성 점수: 성능 높고 워크로드 낮을수록 좋음
            efficiency_score = prov.throughput / (1.0 + current_load)
            providers_by_efficiency.append((provider_id, efficiency_score))
        
        providers_by_efficiency.sort(key=lambda x: x[1], reverse=True)
        
        # 선택된 Provider 중에서 최고 효율성 순으로 할당
        if len(providers_by_efficiency) < task.scene_number:
            # 필요한 수보다 적으면 중복 허용 (다른 씬에 같은 GPU 할당)
            selected_providers = [provider_id for provider_id, _ in providers_by_efficiency]
            while len(selected_providers) < task.scene_number:
                selected_providers.extend([provider_id for provider_id, _ in providers_by_efficiency])
            selected_providers = selected_providers[:task.scene_number]
        else:
            # 1:1 매칭 가능
            selected_providers = [provider_id for provider_id, _ in providers_by_efficiency[:task.scene_number]]
        
        # 큰 씬 -> 높은 효율성 Provider 매칭
        allocation = [0] * task.scene_number
        for i, (scene_id, _) in enumerate(scenes_by_size):
            allocation[scene_id] = selected_providers[i]
        
        return allocation

# ----------------------------------------------------------------
# 3) 통합 Provider 배치 처리 스케줄러
# ----------------------------------------------------------------
class BatchAwareProviderSelector(ProviderSelector):
    """Provider 배치 처리와 동시 최적화를 통합한 ProviderSelector"""
    
    def __init__(self, 
                 batch_size: int = 5,
                 strategy: str = "performance_first",
                 wait_for_provider_batch_complete: bool = False,
                 load_balancing: bool = True):
        """
        Args:
            batch_size: Provider 배치 크기
            strategy: Provider 배치 선택 전략
            wait_for_provider_batch_complete: Provider 배치가 완성될 때까지 기다릴지 여부
            load_balancing: 부하 분산 여부
        """
        self.provider_selector = BatchProviderSelector(batch_size, strategy, True, load_balancing)
        self.provider_optimizer = ConcurrentProviderOptimizer(batch_size)
        self.wait_for_provider_batch_complete = wait_for_provider_batch_complete
        self.pending_providers: List[int] = []
    
    def select_providers(self, 
                        providers: Providers, 
                        sim_time: datetime.datetime,
                        required_count: int = None) -> List[int]:
        """배치 인식 Provider 선택"""
        
        # Provider 배치 선택
        selected_batch = self.provider_selector.select_providers(
            providers, sim_time, required_count
        )
        
        if len(selected_batch) > 1:
            print(f"\n=== Provider 배치 처리 시작 ===")
            print(f"선택된 GPU들: {selected_batch}")
            print(f"배치 크기: {len(selected_batch)}")
            
            # Provider 배치를 동시 최적화 대기란에 추가
            batch_id = f"provider_batch_{sim_time.strftime('%Y%m%d_%H%M%S')}"
            self.provider_optimizer.add_provider_batch(batch_id, selected_batch, providers)
        
        return selected_batch
    
    def get_provider_optimizer(self) -> ConcurrentProviderOptimizer:
        """Provider 배치 최적화기 반환"""
        return self.provider_optimizer

# ----------------------------------------------------------------
# 4) 사용 예시 및 테스트
# ----------------------------------------------------------------
def example_usage():
    """사용 예시"""
    
    # 1. 기본 Provider 배치 처리
    print("=== 기본 배치 ProviderSelector ===")
    provider_selector = BatchProviderSelector(
        batch_size=5, 
        strategy="performance_first",
        load_balancing=True
    )
    
    # 2. 동시 최적화 포함 Provider 배치 처리  
    print("\n=== 동시 최적화 Provider 배치 ProviderSelector ===")
    concurrent_selector = BatchAwareProviderSelector(
        batch_size=5, 
        strategy="cost_first",
        wait_for_provider_batch_complete=False,
        load_balancing=True
    )
    
    print("Provider 배치 처리 모듈이 준비되었습니다!")
    print("사용법:")
    print("1. BatchProviderSelector: 단순 Provider 배치 처리")
    print("2. BatchAwareProviderSelector: 동시 최적화 포함 Provider 배치 처리")
    print("3. 기존 ComboGenerator에서 provider 선택 시 사용 가능")
    print("\n전략 옵션:")
    print("- performance_first: 성능 높은 순")
    print("- cost_first: 비용 낮은 순")
    print("- idle_first: 유휴율 높은 순")
    print("- availability_first: 대역폭 높은 순")
    print("- random: 랜덤 선택")

if __name__ == "__main__":
    example_usage() 