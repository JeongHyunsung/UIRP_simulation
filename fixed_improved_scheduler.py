#!/usr/bin/env python3
"""
수정된 개선 배치 스케줄러
• Provider 생성 오류 수정
• 성능 최적화
• 실제 배정 보장
"""

from __future__ import annotations
import datetime
import time
import threading
from typing import List, Tuple, Optional, Dict, Set
from ortools.sat.python import cp_model
import json
import copy

from tasks import Tasks, Task
from providers import Providers, Provider

# (task_id, scene_id, start, finish, provider_idx)
Assignment = Tuple[str, int, datetime.datetime, datetime.datetime, int]


class Scene:
    def __init__(self, workload: float):
        self.workload = workload


class OptimizedTask:
    def __init__(self, task_id: str, scenes: list, budget: float, deadline: datetime.datetime, priority: int = 2):
        self.task_id = task_id
        self.scenes = scenes
        self.budget = budget
        self.deadline = deadline
        self.priority = priority


class OptimizedTasks:
    def __init__(self):
        self.tasks = {}
    
    def add_task(self, task):
        self.tasks[task.task_id] = task


class ProviderWrapper:
    def __init__(self, provider, name="Provider"):
        self._provider = provider
        self.name = name
        self.gpu_count = int(provider.throughput)
        self.cost_per_hour = provider.price_per_gpu_hour
        if provider.available_hours:
            self.start_time = provider.available_hours[0][0]
            self.end_time = provider.available_hours[0][1]
        else:
            self.start_time = datetime.datetime(2024, 1, 1, 0, 0)
            self.end_time = datetime.datetime(2024, 1, 1, 23, 59)


class ProvidersWrapper:
    def __init__(self, providers):
        self.providers = []
        for i, provider in enumerate(providers):
            wrapped = ProviderWrapper(provider, f"Provider_{i}")
            self.providers.append(wrapped)


class FixedImprovedScheduler:
    """수정된 개선 배치 스케줄러"""

    def __init__(
        self,
        slot_duration_minutes: int = 60,
        weight_throughput: float = 10.0,
        weight_cost: float = 1.0,
        weight_deadline: float = 5.0,
        weight_priority: float = 15.0,
        batch_size: int = 3,
        batch_timeout_seconds: int = 300,
        auto_scale_threshold: float = 50.0,
        verbose: bool = True,
    ):
        self.slot_duration = datetime.timedelta(minutes=slot_duration_minutes)
        self.weights = {
            'throughput': weight_throughput,
            'cost': weight_cost,
            'deadline': weight_deadline,
            'priority': weight_priority,
        }
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout_seconds
        self.auto_scale_threshold = auto_scale_threshold
        self.verbose = verbose
        
        # 상태 관리
        self.tasks: OptimizedTasks = OptimizedTasks()
        self.providers: Providers = Providers()
        self.current_assignments: List[Assignment] = []
        self.locked_assignments: Set[Tuple[str, int]] = set()
        self.current_time: Optional[datetime.datetime] = None
        
        # 배치 처리 관련
        self.batch_queue: List[OptimizedTask] = []
        self.batch_lock = threading.Lock()
        self.last_batch_time: Optional[float] = None
        self.batch_stats = {
            'total_batches': 0,
            'total_tasks_processed': 0,
            'avg_batch_size': 0.0,
            'total_optimization_time': 0.0,
            'avg_success_rate': 0.0
        }

    def initialize(self, tasks: OptimizedTasks, providers: Providers, current_time: datetime.datetime):
        """초기 설정"""
        self.tasks = copy.deepcopy(tasks)
        self.providers = ProvidersWrapper(providers)
        self.current_time = current_time
        return self._optimize_all()

    def add_task(self, task: OptimizedTask, auto_batch: bool = True) -> bool:
        """새 Task 추가"""
        with self.batch_lock:
            self.batch_queue.append(task)
            if self.verbose:
                print(f"✅ Task '{task.task_id}' 배치 큐에 추가됨 (큐 크기: {len(self.batch_queue)}/{self.batch_size})")
            
            if auto_batch and len(self.batch_queue) >= self.batch_size:
                if self.verbose:
                    print(f"🚀 배치 크기 조건 만족 ({len(self.batch_queue)}/{self.batch_size})")
                return self._process_batch()
            
            return True

    def add_provider(self, provider_data: dict, auto_reoptimize: bool = True) -> bool:
        """🔧 수정된 Provider 동적 추가"""
        try:
            # Provider 클래스는 딕셔너리를 받음
            provider_dict = {
                "throughput": provider_data["throughput"],
                "price": provider_data["price"],
                "bandwidth": provider_data["bandwidth"],
                "available_hours": provider_data["available_hours"]
            }
            
            new_provider = Provider(provider_dict)
            wrapped = ProviderWrapper(new_provider, provider_data["name"])
            
            # Providers에 추가
            self.providers.providers.append(wrapped)
            
            if self.verbose:
                print(f"✅ 새 Provider 추가됨: {provider_data['name']}")
                print(f"   • GPU 개수: {provider_data['throughput']}")
                print(f"   • 시간당 비용: ${provider_data['price']}")
                print(f"   • 가용 시간: {provider_data['available_hours'][0][0]} ~ {provider_data['available_hours'][0][1]}")
            
            # 자동 재최적화
            if auto_reoptimize and len(self.tasks.tasks) > 0:
                if self.verbose:
                    print("🔄 새 Provider 추가로 인한 재최적화 실행...")
                return self._optimize_all()
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Provider 추가 실패: {e}")
            return False

    def _process_batch(self) -> bool:
        """배치 처리 실행"""
        if len(self.batch_queue) == 0:
            return True
            
        start_time = time.time()
        batch_tasks = self.batch_queue.copy()
        self.batch_queue.clear()
        self.last_batch_time = time.time()
        
        if self.verbose:
            print(f"\n🎯 배치 처리 시작: {len(batch_tasks)}개 Task")
            for task in batch_tasks:
                priority_text = {1: "높음", 2: "중간", 3: "낮음"}.get(task.priority, "보통")
                print(f"   - {task.task_id}: {len(task.scenes)}개 씬 (우선순위: {priority_text})")
        
        # Task를 시스템에 추가
        for task in batch_tasks:
            self.tasks.add_task(task)
        
        # 전체 재최적화
        success = self._optimize_all()
        
        # 성공률 확인 및 자동 스케일링
        if success:
            summary = self.get_assignment_summary()
            if summary['success_rate'] < self.auto_scale_threshold:
                if self.verbose:
                    print(f"🚨 성공률 {summary['success_rate']:.1f}% 미달 - 자동 스케일링 검토")
                self._suggest_scaling()
        
        # 통계 업데이트
        processing_time = time.time() - start_time
        self._update_batch_stats(len(batch_tasks), processing_time, success)
        
        if self.verbose:
            print(f"⚡ 배치 처리 완료 ({processing_time:.2f}초)")
            self._print_batch_stats()
        
        return success

    def _suggest_scaling(self):
        """스케일링 제안"""
        current_gpu_capacity = sum(p.gpu_count for p in self.providers.providers)
        total_workload = sum(
            sum(scene.workload for scene in task.scenes)
            for task in self.tasks.tasks.values()
        )
        
        if self.verbose:
            print(f"📊 리소스 분석:")
            print(f"   • 현재 GPU 용량: {current_gpu_capacity}")
            print(f"   • 필요 workload: {total_workload}")
            if total_workload > current_gpu_capacity:
                print(f"   • 💡 권장: GPU {int(total_workload - current_gpu_capacity + 2)}개 추가 필요")

    def _optimize_all(self) -> bool:
        """🎯 개선된 최적화 실행"""
        if len(self.tasks.tasks) == 0:
            if self.verbose:
                print("❌ 최적화할 Task가 없습니다")
            return False

        model = cp_model.CpModel()
        cost_scale = 100
        
        # 시간 슬롯 생성
        time_slots = self._generate_time_slots()
        if self.verbose:
            print(f"🕒 시간 슬롯: {len(time_slots)}개 ({time_slots[0]} ~ {time_slots[-1]})")
        
        # 변수 생성 및 씬 정보 수집
        x = {}
        scene_info = []
        
        for task in self.tasks.tasks.values():
            x[task.task_id] = {}
            for scene_id, scene in enumerate(task.scenes):
                priority = getattr(task, 'priority', 2)
                scene_info.append((
                    task.task_id, 
                    scene_id, 
                    scene.workload, 
                    self._get_deadline_slot(task.deadline, time_slots),
                    priority
                ))
                x[task.task_id][scene_id] = {}
                
                for provider_idx in range(len(self.providers.providers)):
                    x[task.task_id][scene_id][provider_idx] = {}
                    for slot_idx in range(len(time_slots)):
                        var_name = f"x_{task.task_id}_{scene_id}_{provider_idx}_{slot_idx}"
                        x[task.task_id][scene_id][provider_idx][slot_idx] = model.NewBoolVar(var_name)

        # 🔧 개선된 제약조건 1: 부분 배정 허용 + 최소 1개 보장
        total_assigned = model.NewIntVar(0, len(scene_info), "total_assigned")
        model.Add(total_assigned == sum(
            x[task_id][scene_id][provider_idx][slot_idx]
            for task_id, scene_id, _, _, _ in scene_info
            for provider_idx in range(len(self.providers.providers))
            for slot_idx in range(len(time_slots))
        ))
        
        # 최소 1개 씬은 배정되도록 보장 (가능한 경우)
        model.Add(total_assigned >= 1)

        for task_id, scene_id, _, _, _ in scene_info:
            if (task_id, scene_id) in self.locked_assignments:
                # 잠긴 배정 처리
                locked_assignment = self._get_locked_assignment(task_id, scene_id)
                if locked_assignment:
                    provider_idx, start_slot = locked_assignment
                    model.Add(x[task_id][scene_id][provider_idx][start_slot] == 1)
                    for p_idx in range(len(self.providers.providers)):
                        for s_idx in range(len(time_slots)):
                            if p_idx != provider_idx or s_idx != start_slot:
                                model.Add(x[task_id][scene_id][p_idx][s_idx] == 0)
                else:
                    for p_idx in range(len(self.providers.providers)):
                        for s_idx in range(len(time_slots)):
                            model.Add(x[task_id][scene_id][p_idx][s_idx] == 0)
            else:
                # 부분 배정 허용
                model.Add(sum(x[task_id][scene_id][provider_idx][slot_idx]
                             for provider_idx in range(len(self.providers.providers))
                             for slot_idx in range(len(time_slots))) <= 1)

        # 제약조건 2: GPU 용량 제약
        for provider_idx, provider in enumerate(self.providers.providers):
            for slot_idx in range(len(time_slots)):
                total_workload = sum(int(workload * cost_scale) * x[task_id][scene_id][provider_idx][slot_idx]
                                   for task_id, scene_id, workload, _, _ in scene_info)
                model.Add(total_workload <= int(provider.gpu_count * cost_scale))

        # 제약조건 3: Provider 가용 시간
        for provider_idx, provider in enumerate(self.providers.providers):
            for slot_idx, slot_time in enumerate(time_slots):
                if slot_time < provider.start_time or slot_time >= provider.end_time:
                    for task_id, scene_id, _, _, _ in scene_info:
                        model.Add(x[task_id][scene_id][provider_idx][slot_idx] == 0)

        # 제약조건 4: 예산 제약 (간소화)
        for task in self.tasks.tasks.values():
            task_scenes = [(task.task_id, scene_id) for scene_id in range(len(task.scenes))]
            total_cost = sum(
                int(self.providers.providers[provider_idx].cost_per_hour * workload * cost_scale) *
                x[task_id][scene_id][provider_idx][slot_idx]
                for task_id, scene_id in task_scenes
                for task_id2, scene_id2, workload, _, _ in scene_info
                if task_id == task_id2 and scene_id == scene_id2
                for provider_idx in range(len(self.providers.providers))
                for slot_idx in range(len(time_slots))
            )
            model.Add(total_cost <= int(task.budget * cost_scale * cost_scale))

        # 🎯 간단한 목적함수: 처리량 최대화
        total_assigned_with_priority = sum(
            (4 - priority) * x[task_id][scene_id][provider_idx][slot_idx]
            for task_id, scene_id, _, _, priority in scene_info
            for provider_idx in range(len(self.providers.providers))
            for slot_idx in range(len(time_slots))
        )

        # 목적함수: 우선순위 가중 처리량 최대화
        model.Maximize(total_assigned_with_priority)

        # 최적화 실행
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            self._extract_solution(solver, x, time_slots, scene_info)
            return True
        else:
            if self.verbose:
                print(f"❌ 최적화 실패: {solver.StatusName(status)}")
            return False

    def _generate_time_slots(self) -> List[datetime.datetime]:
        """시간 슬롯 생성"""
        if self.current_time is None:
            return []
            
        slots = []
        current = self.current_time
        max_slots = 48
        for _ in range(max_slots):
            slots.append(current)
            current += self.slot_duration
            
        return slots

    def _get_deadline_slot(self, deadline: datetime.datetime, time_slots: List[datetime.datetime]) -> int:
        """데드라인에 해당하는 슬롯 인덱스 반환"""
        for i, slot_time in enumerate(time_slots):
            if slot_time >= deadline:
                return max(0, i - 1)
        return len(time_slots) - 1

    def _get_locked_assignment(self, task_id: str, scene_id: int) -> Optional[Tuple[int, int]]:
        """잠긴 배정의 provider_idx, slot_idx 반환"""
        for assignment in self.current_assignments:
            if assignment[0] == task_id and assignment[1] == scene_id:
                provider_idx = assignment[4]
                time_slots = self._generate_time_slots()
                for slot_idx, slot_time in enumerate(time_slots):
                    if slot_time == assignment[2]:
                        return provider_idx, slot_idx
        return None

    def _extract_solution(self, solver, x, time_slots, scene_info):
        """해답 추출 및 배정 업데이트"""
        new_assignments = []
        
        for task_id, scene_id, workload, _, _ in scene_info:
            for provider_idx in range(len(self.providers.providers)):
                for slot_idx in range(len(time_slots)):
                    if solver.Value(x[task_id][scene_id][provider_idx][slot_idx]) == 1:
                        start_time = time_slots[slot_idx]
                        finish_time = start_time + self.slot_duration
                        
                        assignment = (task_id, scene_id, start_time, finish_time, provider_idx)
                        new_assignments.append(assignment)
                        
                        # 현재 실행 중인 작업은 잠금
                        if start_time <= self.current_time < finish_time:
                            self.locked_assignments.add((task_id, scene_id))

        # 새로운 배정으로 교체
        unlocked_assignments = [a for a in new_assignments 
                              if (a[0], a[1]) not in self.locked_assignments]
        locked_assignments = [a for a in self.current_assignments 
                            if (a[0], a[1]) in self.locked_assignments]
        
        self.current_assignments = locked_assignments + unlocked_assignments
        
        if self.verbose:
            self._print_assignments()

    def _print_assignments(self):
        """현재 배정 결과 출력"""
        if not self.current_assignments:
            print("❌ 배정된 씬이 없습니다")
            return

        print(f"\n📋 현재 배정 결과 ({len(self.current_assignments)}개 씬):")
        
        # Task별로 그룹화
        by_task = {}
        total_cost = 0.0
        
        for task_id, scene_id, start, finish, provider_idx in self.current_assignments:
            if task_id not in by_task:
                by_task[task_id] = []
            by_task[task_id].append((scene_id, start, finish, provider_idx))
            
            # 비용 계산
            task = self.tasks.tasks[task_id]
            workload = task.scenes[scene_id].workload
            cost = self.providers.providers[provider_idx].cost_per_hour * workload
            total_cost += cost

        for task_id, scenes in by_task.items():
            task = self.tasks.tasks[task_id]
            priority_text = {1: "높음", 2: "중간", 3: "낮음"}.get(getattr(task, 'priority', 2), "보통")
            print(f"\n🎯 Task '{task_id}' ({len(scenes)}/{len(task.scenes)} 씬 배정) - 우선순위: {priority_text}")
            print(f"   예산: ${task.budget:.2f}, 데드라인: {task.deadline}")
            
            task_cost = 0.0
            for scene_id, start, finish, provider_idx in sorted(scenes):
                provider = self.providers.providers[provider_idx]
                workload = task.scenes[scene_id].workload
                cost = provider.cost_per_hour * workload
                task_cost += cost
                
                lock_status = "🔒" if (task_id, scene_id) in self.locked_assignments else "🔓"
                print(f"   {lock_status} Scene {scene_id}: {start.strftime('%H:%M')}-{finish.strftime('%H:%M')} "
                      f"@ {provider.name} (${cost:.2f})")
            
            success_rate = len(scenes) / len(task.scenes) * 100
            print(f"   💰 Task 비용: ${task_cost:.2f} | 성공률: {success_rate:.1f}%")

        print(f"\n💰 총 비용: ${total_cost:.2f}")
        print(f"🔒 잠긴 배정: {len(self.locked_assignments)}개")

    def _update_batch_stats(self, batch_size: int, processing_time: float, success: bool):
        """배치 처리 통계 업데이트"""
        self.batch_stats['total_batches'] += 1
        self.batch_stats['total_tasks_processed'] += batch_size
        self.batch_stats['total_optimization_time'] += processing_time
        
        total_tasks = self.batch_stats['total_tasks_processed']
        total_batches = self.batch_stats['total_batches']
        self.batch_stats['avg_batch_size'] = total_tasks / total_batches
        
        if success:
            summary = self.get_assignment_summary()
            current_success_rate = summary['success_rate']
            prev_avg = self.batch_stats['avg_success_rate']
            self.batch_stats['avg_success_rate'] = (prev_avg * (total_batches - 1) + current_success_rate) / total_batches

    def _print_batch_stats(self):
        """배치 처리 통계 출력"""
        stats = self.batch_stats
        print(f"📊 배치 처리 통계:")
        print(f"   • 총 배치 수: {stats['total_batches']}")
        print(f"   • 처리된 Task 수: {stats['total_tasks_processed']}")
        print(f"   • 평균 배치 크기: {stats['avg_batch_size']:.1f}")
        print(f"   • 총 최적화 시간: {stats['total_optimization_time']:.2f}초")
        print(f"   • 평균 성공률: {stats['avg_success_rate']:.1f}%")
        if stats['total_batches'] > 0:
            avg_time = stats['total_optimization_time'] / stats['total_batches']
            print(f"   • 평균 배치 처리 시간: {avg_time:.2f}초")

    def update_time(self, new_time: datetime.datetime):
        """시간 업데이트"""
        self.current_time = new_time
        self._update_locked_assignments()

    def _update_locked_assignments(self):
        """실행 중인 작업 확인 및 잠금 상태 업데이트"""
        if self.current_time is None:
            return
            
        completed = set()
        for task_id, scene_id in self.locked_assignments:
            for assignment in self.current_assignments:
                if assignment[0] == task_id and assignment[1] == scene_id:
                    if assignment[3] <= self.current_time:
                        completed.add((task_id, scene_id))
                    break
        
        self.locked_assignments -= completed
        
        if self.verbose and completed:
            print(f"🔓 {len(completed)}개 작업 잠금 해제: {completed}")

    def get_assignment_summary(self) -> Dict:
        """배정 결과 요약 반환"""
        if not self.current_assignments:
            total_scenes = sum(len(task.scenes) for task in self.tasks.tasks.values()) if self.tasks.tasks else 0
            return {
                'total_scenes': total_scenes,
                'assigned_scenes': 0,
                'success_rate': 0.0,
                'total_cost': 0.0,
                'tasks': {},
                'batch_stats': self.batch_stats.copy()
            }

        by_task = {}
        total_cost = 0.0
        
        for task_id, scene_id, start, finish, provider_idx in self.current_assignments:
            if task_id not in by_task:
                by_task[task_id] = []
            by_task[task_id].append((scene_id, start, finish, provider_idx))
            
            task = self.tasks.tasks[task_id]
            workload = task.scenes[scene_id].workload
            cost = self.providers.providers[provider_idx].cost_per_hour * workload
            total_cost += cost

        total_scenes = sum(len(task.scenes) for task in self.tasks.tasks.values())
        assigned_scenes = len(self.current_assignments)
        
        task_summaries = {}
        for task_id in self.tasks.tasks:
            task = self.tasks.tasks[task_id]
            assigned_count = len(by_task.get(task_id, []))
            task_summaries[task_id] = {
                'total_scenes': len(task.scenes),
                'assigned_scenes': assigned_count,
                'success_rate': assigned_count / len(task.scenes) * 100,
                'budget': task.budget,
                'priority': getattr(task, 'priority', 2)
            }

        return {
            'total_scenes': total_scenes,
            'assigned_scenes': assigned_scenes,
            'success_rate': assigned_scenes / total_scenes * 100 if total_scenes > 0 else 0,
            'total_cost': total_cost,
            'tasks': task_summaries,
            'batch_stats': self.batch_stats.copy()
        }

    def get_provider_status(self) -> dict:
        """Provider 상태 반환"""
        status = {
            'total_providers': len(self.providers.providers),
            'providers': []
        }
        
        for i, provider in enumerate(self.providers.providers):
            running_tasks = len([
                a for a in self.current_assignments
                if a[4] == i and a[2] <= self.current_time < a[3]
            ])
            
            total_assigned = len([
                a for a in self.current_assignments if a[4] == i
            ])
            
            provider_info = {
                'index': i,
                'name': provider.name,
                'gpu_count': provider.gpu_count,
                'cost_per_hour': provider.cost_per_hour,
                'available_time': f"{provider.start_time} ~ {provider.end_time}",
                'running_tasks': running_tasks,
                'total_assigned': total_assigned,
                'utilization': f"{running_tasks}/{provider.gpu_count}"
            }
            status['providers'].append(provider_info)
        
        return status 