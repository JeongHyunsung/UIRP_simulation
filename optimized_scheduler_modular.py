# ================================================================
# optimized_scheduler_modular.py
# ================================================================
from __future__ import annotations
import datetime, itertools, math
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Sequence
from itertools import combinations

from tasks import Tasks, Task
from providers import Providers, Provider
from objective import calc_objective, DEFAULT_WEIGHTS

Assignment = Tuple[str, int, datetime.datetime, datetime.datetime, int]

# ----------------------------------------------------------------
# 1) 전략 인터페이스
# ----------------------------------------------------------------
class TaskSelector(ABC):
    @abstractmethod
    def select(self, now: datetime.datetime, waiting: Sequence[Task]) -> List[Task]: ...

class ComboGenerator(ABC):
    @abstractmethod
    def best_combo(
        self, task: Task, providers: Providers, sim_time: datetime.datetime,
        evaluator: "MetricEvaluator", verbose: bool = False,
    ) -> Optional[Tuple[List[int], float, float]]: ...

class MetricEvaluator(ABC):
    @abstractmethod
    def time_cost(self, task: Task, scene_id: int, prov: Provider) -> Tuple[float, float]: ...
    @abstractmethod
    def feasible(self, task: Task, combo: List[int], sim_time: datetime.datetime,
                 providers: Providers) -> Tuple[bool, float, float]: ...
    @abstractmethod
    def efficiency(self, task: Task, t_tot: float, cost: float) -> float: ...

class Dispatcher(ABC):
    @abstractmethod
    def dispatch(
        self, task: Task, combo: List[int], sim_time: datetime.datetime,
        providers: Providers, evaluator: MetricEvaluator, verbose: bool
    ) -> List[Assignment]: ...

# ----------------------------------------------------------------
# 2) 기본 전략 (기존과 동일)
# ----------------------------------------------------------------
class FIFOTaskSelector(TaskSelector):
    def select(self, now, waiting): return list(waiting)

class BaselineEvaluator(MetricEvaluator):
    def __init__(self): self._c: Dict[tuple, float] = {}
    def _t(self, t,s,p):
        k=("tx",t.id,s,p);                                                     # transfer
        if k in self._c: return self._c[k]
        v=float('inf') if (bw:=min(t.bandwidth,p.bandwidth))<=0 else \
           (t.global_file_size+t.scene_size(s))/bw/3600
        self._c[k]=v; return v
    def _cpt(self,t,p):
        k=("cmp",t.id,p)
        if k in self._c: return self._c[k]
        v=float('inf') if p.throughput<=0 else t.scene_workload/p.throughput
        self._c[k]=v; return v
    def time_cost(self,t,s,p):
        d=self._t(t,s,p)+self._cpt(t,p); return d,d*p.price_per_gpu_hour
    def feasible(self,t,cmb,now,ps):
        grouped:Dict[int,List[int]]={}
        for sid,p in enumerate(cmb): grouped.setdefault(p,[]).append(sid)
        t_tot,cost=0.0,0.0
        for p,sids in grouped.items():
            seq=sum(self.time_cost(t,sid,ps[p])[0] for sid in sids)
            if any(math.isinf(self.time_cost(t,sid,ps[p])[0]) for sid in sids): return False,math.inf,math.inf
            t_tot=max(t_tot,seq)
            cost += sum(self.time_cost(t,sid,ps[p])[1] for sid in sids)
        if cost>t.budget: return False,t_tot,cost
        if now+datetime.timedelta(hours=t_tot)>t.deadline: return False,t_tot,cost
        return True,t_tot,cost
    def efficiency(self,t,t_tot,cost):
        return 0.0 if t_tot<=0 or cost<=0 else (t.scene_number*t.scene_workload)/(cost*t_tot)

# ----------------------------------------------------------------
# 3) 최적화된 전략
# ----------------------------------------------------------------
class OptimizedEvaluator(MetricEvaluator):
    """objective.py의 손실함수를 사용하는 최적화된 평가자"""
    
    def __init__(self, weights: Tuple[float, float, float, float, float] = DEFAULT_WEIGHTS):
        self.weights = weights
        self._cache: Dict[tuple, Tuple[float, float]] = {}
    
    def time_cost(self, task: Task, scene_id: int, prov: Provider) -> Tuple[float, float]:
        """개별 씬의 시간과 비용 계산"""
        cache_key = ("time_cost", task.id, scene_id, id(prov))
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 기본 계산
        size = task.global_file_size + task.scene_size(scene_id)
        rate = min(task.bandwidth, prov.bandwidth)
        if rate <= 0 or prov.throughput <= 0:
            result = (float('inf'), float('inf'))
        else:
            T_tx = size / rate / 3600  # 시간 단위로 변환
            T_cmp = task.scene_workload / prov.throughput
            total_time = T_tx + T_cmp
            cost = total_time * prov.price_per_gpu_hour
            result = (total_time, cost)
        
        self._cache[cache_key] = result
        return result
    
    def feasible(self, task: Task, combo: List[int], sim_time: datetime.datetime,
                 providers: Providers) -> Tuple[bool, float, float]:
        """조합의 실행 가능성과 전체 시간, 비용 계산"""
        if len(combo) != task.scene_number:
            return False, float('inf'), float('inf')
        
        scene_times = []
        total_cost = 0.0
        
        for scene_id, gpu_id in enumerate(combo):
            if gpu_id >= len(providers):
                return False, float('inf'), float('inf')
            
            prov = providers[gpu_id]
            time, cost = self.time_cost(task, scene_id, prov)
            
            if math.isinf(time) or math.isinf(cost):
                return False, float('inf'), float('inf')
            
            scene_times.append(time)
            total_cost += cost
        
        # makespan (가장 오래 걸리는 씬의 시간)
        makespan = max(scene_times)
        
        # 제약조건 체크
        budget_violated = total_cost > task.budget
        deadline_hours = (task.deadline - sim_time).total_seconds() / 3600.0
        deadline_violated = makespan > deadline_hours
        
        if budget_violated or deadline_violated:
            if budget_violated and deadline_violated:
                print("제약조건변경필요: 예산 및 시간 데드라인 초과")
            elif budget_violated:
                print("제약조건변경필요: 예산 데드라인 초과")
            else:
                print("제약조건변경필요: 시간 데드라인 초과")
            return False, makespan, total_cost
        
        return True, makespan, total_cost
    
    def efficiency(self, task: Task, t_tot: float, cost: float) -> float:
        """호환성을 위한 효율성 계산 (사용되지 않음)"""
        return 0.0 if t_tot <= 0 or cost <= 0 else (task.scene_number * task.scene_workload) / (cost * t_tot)
    
    def task_objective(self, task: Task, combo: List[int], sim_time: datetime.datetime,
                      providers: Providers) -> float:
        """Task 단위 손실함수 계산"""
        a1, a2, a3, b1, b2 = self.weights
        
        scene_times = []
        scene_costs = []
        total_profit = 0.0
        selected_providers = []
        
        for scene_id, gpu_id in enumerate(combo):
            prov = providers[gpu_id]
            time, cost = self.time_cost(task, scene_id, prov)
            
            if math.isinf(time) or math.isinf(cost):
                return float('inf')
            
            scene_times.append(time)
            scene_costs.append(cost)
            total_profit += cost  # 공급자 수익 = 사용자 비용
            selected_providers.append(prov)
        
        # Task 레벨 지표
        makespan = max(scene_times)
        total_cost = sum(scene_costs)
        deadline_hours = (task.deadline - sim_time).total_seconds() / 3600.0
        
        # GPU 유휴 페널티 (선택된 GPU들의 평균 유휴율)
        gpu_idle_penalty = sum(prov.idle_ratio() for prov in selected_providers) / len(selected_providers)
        
        # 손실함수 계산
        objective_value = (
            a1 * makespan +
            a2 * max(0.0, total_cost - task.budget) +
            a3 * max(0.0, makespan - deadline_hours) +
            b1 * total_profit +  # 음수 가중치로 최소화
            b2 * gpu_idle_penalty
        )
        
        return objective_value

class OptimizedComboGenerator(ComboGenerator):
    """씬 크기와 GPU 성능 기반 최적화된 조합 생성기"""
    
    def __init__(self):
        pass
    
    def best_combo(
        self, task: Task, providers: Providers, sim_time: datetime.datetime,
        evaluator: MetricEvaluator, verbose: bool = False,
    ) -> Optional[Tuple[List[int], float, float]]:
        """최적 GPU 조합 찾기"""
        
        if not isinstance(evaluator, OptimizedEvaluator):
            # 호환성을 위해 기존 방식으로 처리
            return self._fallback_combo(task, providers, sim_time, evaluator, verbose)
        
        # 사용 가능한 GPU 필터링
        available_gpus = []
        for i, prov in enumerate(providers):
            # is_available_at 메서드가 없을 수 있으므로 기본적으로 사용 가능하다고 가정
            try:
                if hasattr(prov, 'is_available_at') and not prov.is_available_at(sim_time):
                    continue
            except:
                pass  # 메서드가 없으면 사용 가능하다고 가정
            available_gpus.append(i)
        
        if len(available_gpus) < task.scene_number:
            if verbose:
                print(f"    사용 가능한 GPU 수({len(available_gpus)})가 씬 수({task.scene_number})보다 적음")
            return None
        
        # 씬을 크기 순으로 정렬 (내림차순)
        scenes_by_size = []
        for scene_id in range(task.scene_number):
            scene_size = task.scene_size(scene_id)
            scenes_by_size.append((scene_id, scene_size))
        scenes_by_size.sort(key=lambda x: x[1], reverse=True)  # 큰 순서
        
        # GPU를 성능 순으로 정렬 (throughput 내림차순)
        gpus_by_performance = []
        for gpu_id in available_gpus:
            prov = providers[gpu_id]
            gpus_by_performance.append((gpu_id, prov.throughput))
        gpus_by_performance.sort(key=lambda x: x[1], reverse=True)  # 높은 순서
        
        best_objective = float('inf')
        best_combo = None
        best_time = None
        best_cost = None
        
        # 모든 GPU 조합 시도 C(available_gpus, scene_number)
        for gpu_indices in combinations(available_gpus, task.scene_number):
            # GPU를 성능 순으로 정렬
            sorted_gpus = sorted(gpu_indices, key=lambda x: providers[x].throughput, reverse=True)
            
            # 큰 씬 → 고성능 GPU 매칭
            combo = [0] * task.scene_number
            for i, (scene_id, _) in enumerate(scenes_by_size):
                combo[scene_id] = sorted_gpus[i]
            
            # 실행 가능성 체크
            feasible, makespan, total_cost = evaluator.feasible(task, combo, sim_time, providers)
            
            if verbose:
                print(f"    ↪ {combo} feasible={feasible} makespan={makespan:.2f}h cost={total_cost:.1f}$")
            
            if not feasible:
                continue
            
            # 손실함수 계산
            objective_value = evaluator.task_objective(task, combo, sim_time, providers)
            
            if verbose:
                print(f"      objective={objective_value:.4f}")
            
            # 최적값 업데이트
            if objective_value < best_objective:
                best_objective = objective_value
                best_combo = combo
                best_time = makespan
                best_cost = total_cost
        
        if best_combo is None:
            return None
        
        return (best_combo, best_time, best_cost)
    
    def _fallback_combo(self, task, providers, sim_time, evaluator, verbose):
        """기존 BaselineEvaluator와의 호환성을 위한 폴백"""
        best = (-1.0, None)
        max_combinations = 81  # 기존 제한 유지
        
        for cmb in itertools.islice(itertools.product(range(len(providers)), repeat=task.scene_number), max_combinations):
            ok, t_tot, cost = evaluator.feasible(task, list(cmb), sim_time, providers)
            if verbose: 
                print(f"    ↪ {cmb} ok={ok} t={t_tot:.2f}h cost={cost:.1f}$")
            if not ok: 
                continue
            eff = evaluator.efficiency(task, t_tot, cost)
            if eff > best[0]: 
                best = (eff, (list(cmb), t_tot, cost))
        
        return None if best[1] is None else best[1]

class BruteForceGenerator(ComboGenerator):
    def __init__(self,max_comb=81): self.max_comb=max_comb
    def best_combo(self,t,ps,now,ev,verbose=False):
        best=(-1.0,None)
        for cmb in itertools.islice(itertools.product(range(len(ps)),repeat=t.scene_number),self.max_comb):
            ok,t_tot,cost=ev.feasible(t,list(cmb),now,ps)
            if verbose: print(f"    ↪ {cmb} ok={ok} t={t_tot:.2f}h cost={cost:.1f}$")
            if not ok: continue
            eff=ev.efficiency(t,t_tot,cost)
            if eff>best[0]: best=(eff,(list(cmb),t_tot,cost))
        return None if best[1] is None else best[1]

class SequentialDispatcher(Dispatcher):
    def dispatch(self,t,cmb,now,ps,ev,verbose):
        out:List[Assignment]=[]; groups:Dict[int,List[int]]={}
        for sid,p in enumerate(cmb): groups.setdefault(p,[]).append(sid)
        for p,sids in groups.items():
            prov,cur=ps[p],now
            for sid in sids:
                dur,_=ev.time_cost(t,sid,prov); st=prov.earliest_available(dur,cur)
                if st is None: continue
                prov.assign(t.id,sid,st,dur)
                t.scene_allocation_data[sid]=(st,p)
                ft=st+datetime.timedelta(hours=dur)
                out.append((t.id,sid,st,ft,p))
                if verbose: print(f"      scene{sid}->{p} {st.strftime('%m-%d %H:%M')} dur={dur:.2f}")
                cur=ft
        return out

# ----------------------------------------------------------------
# 4) OR-Tools 전략 (선택)
# ----------------------------------------------------------------
try:
    from ortools.sat.python import cp_model
    class CPSatComboGenerator(ComboGenerator):
        def best_combo(self,t,ps,now,ev,verbose=False):
            m=cp_model.CpModel();S,P=t.scene_number,len(ps)
            x=[[m.NewBoolVar(f"x{s}_{p}") for p in range(P)] for s in range(S)]
            for s in range(S): m.Add(sum(x[s])==1)

            big=10**7; dur=[[0]*P for _ in range(S)]; cost=[[0]*P for _ in range(S)]
            for s in range(S):
                for p in range(P):
                    d,c=ev.time_cost(t,s,ps[p])
                    dur[s][p]=big if math.isinf(d) else int(d*3600)
                    cost[s][p]=big if math.isinf(d) else int(c*100)
            makespan=m.NewIntVar(0,big,"mk")
            for p in range(P):
                tot=m.NewIntVar(0,big,f"tot{p}")
                m.Add(tot==sum(dur[s][p]*x[s][p] for s in range(S))); m.Add(tot<=makespan)
            total_cost=sum(cost[s][p]*x[s][p] for s in range(S) for p in range(P))
            m.Add(total_cost<=int(t.budget*100))
            m.Add(makespan<=int((t.deadline-now).total_seconds()))
            m.Minimize(makespan)

            slv=cp_model.CpSolver()
            if verbose: slv.parameters.log_search_progress=True
            slv.parameters.max_time_in_seconds=5
            st=slv.Solve(m)
            if st not in (cp_model.OPTIMAL,cp_model.FEASIBLE): return None
            cmb=[next(p for p in range(P) if slv.BooleanValue(x[s][p])) for s in range(S)]
            return cmb,slv.Value(makespan)/3600,slv.Value(total_cost)/100
    class CPSatDispatcher(SequentialDispatcher): pass
except ImportError:
    CPSatComboGenerator=CPSatDispatcher=None

# ----------------------------------------------------------------
# 5) 레지스트리 & Scheduler
# ----------------------------------------------------------------
COMBO_REG={"bf":BruteForceGenerator, "optimized": OptimizedComboGenerator}
DISP_REG ={"bf":SequentialDispatcher, "optimized": SequentialDispatcher}
if CPSatComboGenerator: COMBO_REG["cp"]=CPSatComboGenerator; DISP_REG["cp"]=CPSatDispatcher

class OptimizedScheduler:
    def __init__(self,*,algo="optimized",time_gap=datetime.timedelta(hours=1),
                 selector:TaskSelector=FIFOTaskSelector(),
                 evaluator:MetricEvaluator=None,
                 verbose=False):
        self.selector=selector
        self.generator=COMBO_REG[algo]()
        self.dispatcher=DISP_REG[algo]()
        self.evaluator=evaluator if evaluator is not None else OptimizedEvaluator()
        self.time_gap=time_gap
        self.verbose=verbose
        self.waiting_tasks:List[Task]=[]
        self.results:List[Assignment]=[]

    def _feed(self,now,tasks):
        ids={t.id for t in self.waiting_tasks}
        for t in tasks:
            if t.start_time<=now and t.id not in ids and any(st is None for st,_ in t.scene_allocation_data):
                self.waiting_tasks.append(t)
        self.waiting_tasks.sort(key=lambda t:t.start_time)

    def _schedule_once(self,now,ps):
        new:List[Assignment]=[]; remain=[]
        for t in self.selector.select(now,self.waiting_tasks):
            if not all(st is None for st,_ in t.scene_allocation_data):
                remain.append(t); continue
            best=self.generator.best_combo(t,ps,now,self.evaluator,
                                           verbose=self.verbose)
            if best is None: remain.append(t); continue
            cmb,t_tot,cost=best
            if self.verbose: print(f"[{t.id}] choose {cmb} t={t_tot:.2f}h cost={cost:.1f}$")
            new+=self.dispatcher.dispatch(t,cmb,now,ps,self.evaluator,self.verbose)
        self.waiting_tasks=remain; return new

    def run(self,tasks:Tasks,ps:Providers,
            time_start:datetime.datetime|None=None,
            time_end:datetime.datetime|None=None)->List[Assignment]:
        if time_start is None:
            time_start=min(min(p.available_hours)[0] for p in ps)
        now=time_start
        if time_end is None:
            time_end=max(t.deadline for t in tasks)+datetime.timedelta(days=1)
        while now<time_end:
            self._feed(now,tasks)
            self.results+=self._schedule_once(now,ps)
            if all(all(st is not None for st,_ in t.scene_allocation_data) for t in tasks):
                break
            now+=self.time_gap
        return self.results

# 기존 BaselineScheduler도 유지 (호환성)
class BaselineScheduler:
    def __init__(self,*,algo="bf",time_gap=datetime.timedelta(hours=1),
                 selector:TaskSelector=FIFOTaskSelector(),
                 evaluator:MetricEvaluator=BaselineEvaluator(),
                 verbose=False):
        self.selector=selector
        self.generator=COMBO_REG[algo]()
        self.dispatcher=DISP_REG[algo]()
        self.evaluator=evaluator
        self.time_gap=time_gap
        self.verbose=verbose
        self.waiting_tasks:List[Task]=[]
        self.results:List[Assignment]=[]

    def _feed(self,now,tasks):
        ids={t.id for t in self.waiting_tasks}
        for t in tasks:
            if t.start_time<=now and t.id not in ids and any(st is None for st,_ in t.scene_allocation_data):
                self.waiting_tasks.append(t)
        self.waiting_tasks.sort(key=lambda t:t.start_time)

    def _schedule_once(self,now,ps):
        new:List[Assignment]=[]; remain=[]
        for t in self.selector.select(now,self.waiting_tasks):
            if not all(st is None for st,_ in t.scene_allocation_data):
                remain.append(t); continue
            best=self.generator.best_combo(t,ps,now,self.evaluator,
                                           verbose=self.verbose)
            if best is None: remain.append(t); continue
            cmb,t_tot,cost=best
            if self.verbose: print(f"[{t.id}] choose {cmb} t={t_tot:.2f}h cost={cost:.1f}$")
            new+=self.dispatcher.dispatch(t,cmb,now,ps,self.evaluator,self.verbose)
        self.waiting_tasks=remain; return new

    def run(self,tasks:Tasks,ps:Providers,
            time_start:datetime.datetime|None=None,
            time_end:datetime.datetime|None=None)->List[Assignment]:
        if time_start is None:
            time_start=min(min(p.available_hours)[0] for p in ps)
        now=time_start
        if time_end is None:
            time_end=max(t.deadline for t in tasks)+datetime.timedelta(days=1)
        while now<time_end:
            self._feed(now,tasks)
            self.results+=self._schedule_once(now,ps)
            if all(all(st is not None for st,_ in t.scene_allocation_data) for t in tasks):
                break
            now+=self.time_gap
        return self.results 