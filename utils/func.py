from __future__ import annotations

import re, os, math, time
from rdkit import Chem
import numpy as np
import pandas as pd
from functools import reduce
import gurobipy as gp
from gurobipy import GRB
from time import strftime, gmtime, perf_counter
import threading
import multiprocessing
from multiprocessing import Process, Pool
import cobra

from tqdm import tqdm
from ThermoInfer.utils.constants import *


RT = R * default_T
ln10 = math.log(10)

# Fixed solver settings, kept in sync with the COPT implementation
# (utils/copt_func.py):
#   MIPGap = 1e-3, set per solve in infer_v_range / infer_dGr_range
#   Threads = 1: parallelism is done at the process level. MIP thread
#   scaling is inefficient, and many workers x auto threads oversubscribe
#   the machine. Scale throughput via --processes instead.
_REL_GAP = 1e-3
_THREADS_PER_WORKER = 1

# Deterministic per-solve effort cap, in work units (machine-independent,
# ~1 second of single-thread work on a reference machine). WorkLimit is
# preferred over NodeLimit/TimeLimit: it also counts root LP, presolve, cuts
# and heuristics, so it still terminates solves stuck at the root node where
# NodeLimit is blind, and unlike TimeLimit it does not make the result depend
# on machine speed. A solve hitting the limit marks its reaction as failed:
# it is recorded in <results>_failed.csv instead of the main results file,
# and the user can re-run with a higher limit (via tGEM.work_limit) to
# solve more of them at the cost of longer runtime.
_WORK_LIMIT = 400.0

# statuses meaning "solver stopped early on an effort limit": such solves have
# no usable answer, so the reaction is routed to _failed.csv (not recorded as
# NaN) and is retried automatically if a later run uses a higher limit
_LIMIT_STATUSES = (GRB.WORK_LIMIT, GRB.TIME_LIMIT, GRB.NODE_LIMIT,
                   GRB.SOLUTION_LIMIT, GRB.ITERATION_LIMIT, GRB.INTERRUPTED)
_LIMIT_HIT = object()


def TFBA(model:cobra.core.model.Model,
         thermo_constrain:np.ndarray = None,
         concentration_ub:float = None,
         biomass_synthesis:float = None,
         abs_v_sum:float = None,
         env=None,
         work_limit:float = None) -> dict:
    m = gp.Model(name=f'{model.id} TFBA model', env=env)
    output = {'model':m}
    output['work_limit'] = _WORK_LIMIT if work_limit is None else float(work_limit)
    m.setParam('OutputFlag', 0)
    m.setParam('MIPFocus', 0)
    m.setParam('IntFeasTol', 1e-9)
    m.setParam('FeasibilityTol', 1e-9)
    m.setParam('Threads', _THREADS_PER_WORKER)

    ''' Parameters of the network '''
    lv, uv = np.array([(rxn.lower_bound, rxn.upper_bound) for rxn in model.reactions]).T
    v = m.addMVar(shape=len(model.reactions), vtype='C', lb=lv, ub=uv, name='v')
    output['v'] = v

    ''' Flux balance '''
    S = cobra.util.array.create_stoichiometric_matrix(model) # shape = [met, rxn]
    m.addConstr(S @ v == 0, name="fb")

    ''' Thermodynamic constraints: make sure obey the law '''
    if thermo_constrain is not None:
        try:
            lz, uz = np.array([(met.lz, met.uz) for met in model.metabolites]).T
        except:
            lz, uz = -9, -1
        z = m.addMVar(shape=len(model.metabolites), vtype='C', lb=lz, ub=uz, name='z')
        output['z'] = z
        
        # std are added for loose the constraint
        transformed_standard_dGr_prime = thermo_constrain[:,0]
        transformed_standard_dGr_prime_std = thermo_constrain[:,1]
        std_dGr_uncertainty = transformed_standard_dGr_prime_std * 3 + 5

        have_dGr = (~np.isnan(transformed_standard_dGr_prime))
        std_dGf = m.addMVar(shape=z.shape, vtype='C', lb=-np.inf, ub=np.inf, name='std_dGf')
        output['std_dGf'] = std_dGf

        real_std_dGr = S.T @ std_dGf
        output['real_std_dGr'] = real_std_dGr

        std_dGr_error = transformed_standard_dGr_prime - real_std_dGr
        output['std_dGr_error'] = std_dGr_error

        m.addConstr(std_dGr_error[have_dGr] >= -std_dGr_uncertainty[have_dGr])
        m.addConstr(std_dGr_error[have_dGr] <= std_dGr_uncertainty[have_dGr])
        
        real_dGr = real_std_dGr + RT * ln10 * S.T @ z
        output['real_dGr'] = real_dGr

        # add instrumental variable a0 and constant M0 to linearize to constraint
        # make sure that dGr and v have contrary signs
        # when u_dGr>=0 and l_dGr<=0, then a0=[0 or 1], v is free
        # elif l_dGr<=u_dGr<=0, then a0=1, which means v>=0
        # elif 0<=l_dGr<=u_dGr, then a0=0, which means v<=0
        a0 = m.addMVar(shape=v.shape, vtype='B', name='a0')
        output['a0'] = a0
        
        m.addConstr(v >= -1000 * (1-a0), name='v0') # when 0<=v<=1000, a0=1
        m.addConstr(v <= 1000 * a0, name='v1') # when -1000<=v<=0, a0=0

        m.addConstr(real_dGr[have_dGr] >= -4000 * a0[have_dGr] + 0.0001, name='therb') # when a0=0, v<=0, u_dGr>=0
        m.addConstr(real_dGr[have_dGr] <= 4000 * (1-a0)[have_dGr] - 0.0001, name='therf') # when a0=1, v>=0, l_dGr<=0

        # a reaction's dGr range is bounded iff all its metabolites are anchored
        # by at least one dG-constrained reaction (mirrors utils/copt_func.py)
        anchored_met = np.abs(S[:, have_dGr]).sum(axis=1) > 0
        col_nz = [np.nonzero(S[:, j])[0] for j in range(v.shape[0])]
        anchored = np.array([len(idx) > 0 and bool(anchored_met[idx].all()) for idx in col_nz])
        output['anchored'] = anchored
        

    ''' upper limit of whole metabolic concentration'''
    if concentration_ub:
        intra_met = np.array([met.formula!= 'H2O' and met.compartment != 'e' for met in model.metabolites])
        compartments = [compartment for compartment in model.compartments.keys() if compartment != 'e']
        a10 = m.addMVar(shape=sum(intra_met), vtype='B', name='a10')
        output['a10'] = a10
        a11 = m.addMVar(shape=sum(intra_met), vtype='B', name='a11')
        output['a11'] = a11
        a12 = m.addMVar(shape=sum(intra_met), vtype='B', name='a12')
        output['a12'] = a12
        m.addConstr(z[intra_met] >= (-2*a10) + (-3*a11) + (-11*a12), name='concentration0')
        m.addConstr(z[intra_met] <= (-1*a10) + (-2*a11) + (-3*a12), name='concentration1')
        m.addConstr(a10 + a11 + a12 == 1, name='ivar1')

        concen = (0.09*a10 + 0.009*a11 + (0.001-1e-9)/6*a12) * z[intra_met] + (0.19*a10 + 0.028*a11 + (0.0015-5e-10)*a12)
        output['concentration'] = concen

        for compartment in compartments:
            single_compartment = np.array([met.compartment==compartment for met in model.metabolites])[intra_met]
            sum_concen = gp.quicksum(concen[single_compartment])
            output[f'{compartment}_c'] = concen[single_compartment]
            output[f'{compartment}_z'] = z[intra_met][single_compartment]
            m.addConstr(sum_concen <= concentration_ub, name=f'{compartment}_ub')
            output[f'{compartment}_ub'] = sum_concen
    
    ''' biomass synthesis is necessary'''
    c = np.array([rxn.objective_coefficient for rxn in model.reactions])
    biomass_v = v[c!=0]
    output['biomass_v'] = biomass_v
    if biomass_synthesis is not None:
        m.addConstr(biomass_v >= biomass_synthesis, name='biomass')
    
    '''abs v sum'''
    abs_v = m.addMVar(shape=v.shape, vtype='C', name='abs_v', lb=0, ub=GRB.INFINITY)
    m.addConstr(abs_v >= v, name='abs_v_1')
    m.addConstr(abs_v >= -v, name='abs_v_2')
    output['abs_v'] = abs_v
    if abs_v_sum is not None:
        m.addConstr(gp.quicksum(abs_v) <= abs_v_sum, name='abs_v_sum')

    return output



def infer_v_range(model_dict, v_num, sense='min', OutputFlag=0):
    # main func
    model = model_dict['model']
    model.setParam('OutputFlag', OutputFlag)
    model.setParam('MIPGap', _REL_GAP)
    model.setParam('WorkLimit', model_dict['work_limit'])
    rxn_v = model_dict['v'][v_num]

    model.setObjective(rxn_v, {'min':GRB.MINIMIZE, 'max':GRB.MAXIMIZE}[sense])
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        r = 0.0 if model.ObjVal==-0.0 else np.round(model.ObjVal, 6)
        return r
    if model.Status in (GRB.INF_OR_UNBD, GRB.UNBOUNDED):
        return -np.inf if sense == 'min' else np.inf
    if model.Status in _LIMIT_STATUSES:
        return _LIMIT_HIT
    return np.nan


def infer_dGr_range(model_dict, v_num, sense='min', OutputFlag=0):
    # main func
    model = model_dict['model']
    model.setParam('OutputFlag', OutputFlag)
    model.setParam('MIPGap', _REL_GAP)
    model.setParam('WorkLimit', model_dict['work_limit'])
    dgr = model_dict['real_dGr'][v_num]

    model.setObjective(dgr, {'min':GRB.MINIMIZE, 'max':GRB.MAXIMIZE}[sense])
    model.optimize()
    if model.Status in (GRB.INF_OR_UNBD, GRB.UNBOUNDED):
        return -np.inf if sense == 'min' else np.inf
    if model.Status != GRB.OPTIMAL:
        if not model_dict['anchored'][v_num]:
            return -np.inf if sense == 'min' else np.inf
        if model.Status in _LIMIT_STATUSES:
            return _LIMIT_HIT
        return np.nan
    r = 0.0 if model.ObjVal==-0.0 else np.round(model.ObjVal, 6)
    r = np.inf if r>=1e5 else r
    r = -np.inf if r<=-1e5 else r

    return r



def direction_for_v_range(lv, uv, tol=1e-6):
    if np.isnan(lv) or np.isnan(uv):
        return 'undetermined'
    lv = 0 if abs(lv) <= tol else lv
    uv = 0 if abs(uv) <= tol else uv
    if lv==0 and uv==0:
        return 'blocked'
    elif lv<0 and uv>0:
        return 'bidirectional'
    elif lv<0 and uv<=0:
        return 'backward'
    elif lv>=0 and uv>0:
        return 'forward'
    else:
        print(lv, uv)
        raise ValueError()


def direction_for_dGr_range(ldGr, udGr, tol=1e-6):
    if np.isnan(ldGr) or np.isnan(udGr):
        return 'undetermined'
    ldGr = 0 if abs(ldGr) <= tol else ldGr
    udGr = 0 if abs(udGr) <= tol else udGr
    if ldGr<0 and udGr>0:
        return 'bidirectional'
    elif ldGr>=0 and udGr>0:
        return 'backward'
    elif ldGr<0 and udGr<=0:
        return 'forward'
    else:
        print(ldGr, udGr)
        raise ValueError()


def compare_directions(D1, D2):
    link = {'bidirectional':3,
            'forward':2,
            'backward':2,
            'blocked':0,}
    def compare_d(d1,d2):
        if d1==d2:
            return '='
        else:
            d1 = link[d1]
            d2 = link[d2]
            if d1>d2:
                return '>'
            elif d1<d2:
                return '<'
            elif d1==d2:
                return '!'
            else:
                raise ValueError()
    C = []
    for d1, d2 in zip(D1, D2):
        d = compare_d(d1,d2)
        C.append(d)
    return np.array(C)





def infer_z_range(model, met_num, single_met_exist=True, OutputFlage=0):
    t0 = time.perf_counter()
    # main func
    model = model.copy()
    model.setParam('OutputFlag', 0)
    z = model.getVarByName('z['+str(met_num)+']')
    if single_met_exist:
        fb_met = model.getRow(model.getConstrByName('fb['+str(met_num)+']'))
        met_v = gp.MVar.fromlist([fb_met.getVar(i) for i in range(fb_met.size())])
        a20 = model.addMVar(shape=fb_met.size(), vtype='B', name='a20')
        a21 = model.addVar(vtype='B', name='a21')
        a22 = model.addVar(vtype='B', name='a22')
        sv = sum(met_v * a20)
        model.addConstr(sv >= 0.1 * a21 - 1000 * a22, name='lsv') # LB of sv
        model.addConstr(sv <= 1000 * a21 - 0.1 * a22, name='usv') # UB of sv
        model.addConstr(gp.quicksum(a20) == 1, name='sum_a20')
        model.addLConstr(a21 + a22, GRB.EQUAL, 1, "c0")

    r = []
    for sense in [GRB.MINIMIZE, GRB.MAXIMIZE, ]: #
        model.setObjective(z, sense)
        try:
            model.optimize()
            r.append(model.ObjVal)
        except: # if there is no feasible solution, r = [None, None]
            r = [None, None]
            break
        try:
            assert max(abs(met_v.X))>=0.1*0.5, (z.VarName, met_v.X)
        except:
            r = [False, False]
            break
    
    t1 = time.perf_counter()
    t = strftime("%M:%S", gmtime(t1-t0))
    print(z.VarName, [z.LB, z.UB], '->', r, t)
    return r





def infer_uz(reaction:Reaction, standard_dg=None, nan_to_num=-1000):
    # 
    changed_compound = set()

    if not reaction.is_balanced():
        return changed_compound
    if standard_dg is None:
        standard_dg = reaction.transformed_standard_dGr_prime
    if np.isnan(standard_dg):
        standard_dg = nan_to_num

    for c,_ in reaction.reaction.items():
        if c.Smiles=='[H]O[H]':
            c.lz = H2O_lz
            c.uz = H2O_uz
        elif c.Smiles=='[H+]':
            c.lz = -8.5
        elif c.lz is None:
            print('name', c.name)

    s_uQ = [c.uz * coeff if c.uz is not None else None for c, coeff in reaction.substrates.items()]
    p_lQ = [c.lz * coeff for c, coeff in reaction.products.items()]

    if None not in s_uQ:
        y = standard_dg / RT / math.log(10) + sum(s_uQ) + sum(p_lQ)
        for comp, coeff in reaction.products.items():
            if comp.compartment != 'e':
                uz = np.around(comp.lz - y / coeff, 2)
                uz = min(default_uz, uz)
                if uz < default_lz:
                    continue
                if (comp.uz is None) or (uz > comp.uz):
                    comp.uz = uz
                    changed_compound.add(comp)

    p_uQ = [c.uz * -coeff if c.uz is not None else None for c, coeff in reaction.products.items()]
    s_lQ = [c.lz * -coeff for c, coeff in reaction.substrates.items()]

    if None not in p_uQ:
        y = -standard_dg / RT / math.log(10) + sum(p_uQ) + sum(s_lQ)
        for comp, coeff in reaction.substrates.items():
            if comp.compartment != 'e':
                uz = np.around(comp.lz - y / -coeff, 2)
                uz = min(default_uz, uz)
                if uz < default_lz:
                    continue
                if (comp.uz is None) or (uz > comp.uz):
                    comp.uz = uz
                    changed_compound.add(comp)

    return changed_compound



def can_infer_uz(reaction:Reaction):
    forward = True
    backward = True
    if reaction.is_balanced():
        for compound, coeff in reaction.reaction.items():
            if coeff < 0:
                if compound.uz is None:
                    forward = False
            elif coeff > 0:
                if compound.uz is None:
                    backward = False

    else:
        forward = False
        backward = False
    return (forward, backward)




def infer_conditional_direction(reaction:Reaction, standard_dg=None, toleration=0):
    # 
    if standard_dg is None:
        standard_dg = reaction.transformed_standard_dGr_prime

    if not reaction.is_balanced():
        forward_l_dg = None
        backward_l_dg = None
        forward = None
        backward = None
    elif np.isnan(standard_dg):
        forward_l_dg = np.nan
        backward_l_dg = np.nan
        forward = np.nan
        backward = np.nan
    else:
        forward_Q = sum(c.lz * v for c, v in reaction.products.items()) + sum(c.uz * v if c.uz!=None else default_uz * v for c, v in reaction.substrates.items())
        forward_l_dg = standard_dg + RT * ln10 * forward_Q
        
        backward_Q = -sum(c.lz * v for c, v in reaction.substrates.items()) - sum(c.uz * v if c.uz!=None else default_uz * v for c, v in reaction.products.items())
        backward_l_dg = -standard_dg + RT * ln10 * backward_Q
        
        forward_l_dg = np.round(forward_l_dg, 3)
        backward_l_dg = np.round(backward_l_dg, 3)

        forward = bool(forward_l_dg < -abs(toleration))
        backward = bool(backward_l_dg < -abs(toleration))
            

    reaction.conditional_forward_l_dg = forward_l_dg
    reaction.conditional_backward_l_dg = backward_l_dg

    reaction.conditional_forward = forward
    reaction.conditional_backward = backward

    return [(forward_l_dg, backward_l_dg), (forward, backward)]




def infer_direction(reaction:Reaction, standard_dg=None, toleration=0):
    # 
    if standard_dg is None:
        standard_dg = reaction.transformed_standard_dGr_prime

    if not reaction.is_balanced():
        forward_l_dg = None
        backward_l_dg = None
        forward = None
        backward = None
    elif np.isnan(standard_dg):
        forward_l_dg = np.nan
        backward_l_dg = np.nan
        forward = np.nan
        backward = np.nan
    else:
        forward_l_dg = standard_dg + RT * ln10 * (default_lz * sum(reaction.products.values()) + default_uz * sum(reaction.substrates.values()))
        backward_l_dg = -standard_dg - RT * ln10 * (default_uz * sum(reaction.products.values()) + default_lz * sum(reaction.substrates.values()))
        
        forward_l_dg = np.around(forward_l_dg, 3)
        backward_l_dg = np.around(backward_l_dg, 3)

        forward = bool(forward_l_dg < -abs(toleration))
        backward = bool(backward_l_dg < -abs(toleration))
            

    reaction.forward_l_dg = forward_l_dg
    reaction.backward_l_dg = backward_l_dg

    reaction.forward = forward
    reaction.backward = backward

    return [(forward_l_dg, backward_l_dg), (forward, backward)]




def diff_atom(reaction: Reaction, ignore_H_ion=False, ignore_H2O=False):
    diff_atom = {}
    for comp, coeff in reaction.items():
        if comp.atom_bag is None:
            return None
        for atom, num in comp.atom_bag.items():
            diff_atom[atom] = diff_atom.get(atom, 0) + coeff * num

    if ignore_H_ion:
        diff_atom['H'] = diff_atom.get('H', 0) - diff_atom.get('charge', 0)
        diff_atom['charge'] = 0
    if ignore_H2O:
        diff_atom['H'] = diff_atom.get('H', 0) - 2 * diff_atom.get('O', 0)
        diff_atom['O'] = 0

    
    unbalanced_atom = {}
    for atom, num in diff_atom.items():
        if num!=0:
            unbalanced_atom[atom] = num

    return unbalanced_atom if unbalanced_atom else True


# -----------------------------------------------------------------------
# multiprocessing worker plumbing (mirrors utils/copt_func.py): the tGEM
# instance is passed to each worker exactly once via the pool initializer;
# every worker then builds its model once per chunk and re-uses it for all
# reactions in that chunk.
# -----------------------------------------------------------------------

# Explicit 'fork' context: shared multiprocessing.Value counters only work
# through inheritance; Python >= 3.14 defaults to forkserver on Linux.
_MP_CTX = multiprocessing.get_context('fork')

_worker_tgem = None
_worker_counter = None


def _init_worker(tgem, counter=None):
    global _worker_tgem, _worker_counter
    _worker_tgem = tgem
    _worker_counter = counter


def _run_tfba_chunk(vi_list):
    return vi_list, _worker_tgem.infer_v_and_dGr_batch(vi_list, progress=_worker_counter)


def _run_fba_chunk(vi_list):
    return vi_list, _worker_tgem.infer_v_batch(vi_list, progress=_worker_counter)


def _chunk_size(n_todo, process, batch_size):
    """Task granularity: keep every worker busy (>= 4 tasks per worker),
    but never larger than batch_size. Results are saved per task."""
    if n_todo <= 0:
        return 1
    return max(1, min(batch_size, math.ceil(n_todo / max(1, process * 4))))


def _save_rows(res_path, rows, columns):
    if not rows:
        return
    new_df = pd.DataFrame(data=rows, columns=columns).set_index('rxn num')
    old_df = pd.read_csv(res_path, index_col=0)
    df = new_df if old_df.empty else pd.concat([old_df, new_df], axis=0)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df.to_csv(res_path)


def _bump(progress):
    if progress is not None:
        with progress.get_lock():
            progress.value += 1


def _progress_pump(pbar, counter, stop_event, interval=5):
    last = 0
    while not stop_event.wait(interval):
        with counter.get_lock():
            cur = counter.value
        if cur > last:
            pbar.update(cur - last)
            last = cur
    with counter.get_lock():
        cur = counter.value
    if cur > last:
        pbar.update(cur - last)


class tGEM(object):
    def __init__(self, GEM, dGr, concentration_ub=None, biomass_synthesis=None):
        self.GEM = GEM
        self.dGr = dGr
        self.concentration_ub = concentration_ub
        self.biomass_synthesis = biomass_synthesis
        self.work_limit = None   # None -> module default _WORK_LIMIT

        self.MIPFocus = 0
        self.FBA_res_file_path = None #
        self.TFBA_res_file_path = None # 
    
    def max_biomass_v(self, thermo_constrain=True):
        ''' mutiprocessing the main fun '''
        thermo_constrain = self.dGr if thermo_constrain else None
        t0 = time.perf_counter()
        m = TFBA(self.GEM, thermo_constrain=thermo_constrain, concentration_ub=self.concentration_ub,
                    biomass_synthesis=None, env=None, work_limit=self.work_limit)
        v = m['biomass_v']
        m['model'].setParam('MIPFocus', self.MIPFocus)
        m['model'].setParam('WorkLimit', m['work_limit'])
        m['model'].setObjective(v, GRB.MAXIMIZE)
        m['model'].optimize()

        t1 = time.perf_counter()
        t = strftime("%M:%S", gmtime(t1-t0))
        print(f'Max biomass v: {v.X}', t)
        return v.X
    
    def infer_v(self, vi):
        ''' main func for FBA directionality (builds a one-shot model) '''
        return self.infer_v_batch([vi])[0][1:]

    def infer_v_batch(self, vi_list, progress=None):
        ''' FBA directionality for a list of reactions, one shared LP model '''
        results = []
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        try:
            m = TFBA(self.GEM, thermo_constrain=None, concentration_ub=self.concentration_ub,
                     biomass_synthesis=self.biomass_synthesis, env=env,
                     work_limit=self.work_limit)
            m['model'].setParam('MIPFocus', self.MIPFocus)
            for vi in vi_list:
                try:
                    max_v = infer_v_range(m, vi, 'max')
                    min_v = infer_v_range(m, vi, 'min')
                    if max_v is _LIMIT_HIT or min_v is _LIMIT_HIT:
                        pass
                    elif min_v > max_v:   # NaN passes: only a definite ordering violation fails
                        pass
                    else:
                        results.append((vi, min_v, max_v))
                except Exception as e:
                    print(f'rxn {vi} FBA failed: {e}', flush=True)
                finally:
                    _bump(progress)
        finally:
            env.dispose()

        return results

    def concurrent_infer_v(self, v_si, v_ei=None, process=16, chunk_size=200):
        v_ei = len(self.GEM.reactions) - 1 if v_ei is None else v_ei
        v_ei = min(v_ei, len(self.GEM.reactions) - 1)
        if self.FBA_res_file_path is None:
            print('Please specify the file path of FBA result')
            return None
        elif not os.path.isfile(self.FBA_res_file_path):
            pd.DataFrame(columns=['rxn num', 'lv', 'uv']).to_csv(self.FBA_res_file_path, index=False)

        completed = set(pd.read_csv(self.FBA_res_file_path, index_col=0).index)
        todo = [i for i in range(v_si, v_ei + 1)
                if (not self.GEM.reactions[i].boundary) and (i not in completed)]
        if not todo:
            print('All done')
            return None

        chunk = _chunk_size(len(todo), process, chunk_size)
        chunks = [todo[k:k + chunk] for k in range(0, len(todo), chunk)]
        counter = _MP_CTX.Value('i', 0)
        failed = []
        with _MP_CTX.Pool(process, initializer=_init_worker, initargs=(self, counter)) as pool:
            pbar = tqdm(total=len(todo), desc='Gurobi-FBA inference')
            stop_event = threading.Event()
            pump = threading.Thread(target=_progress_pump, args=(pbar, counter, stop_event), daemon=True)
            pump.start()
            try:
                for vi_list, res in pool.imap_unordered(_run_fba_chunk, chunks):
                    done_ids = {r[0] for r in res}
                    failed.extend(vi for vi in vi_list if vi not in done_ids)
                    _save_rows(self.FBA_res_file_path, res, ['rxn num', 'lv', 'uv'])
            finally:
                stop_event.set()
                pump.join()
                pbar.close()

        if failed:
            failed_path = self.FBA_res_file_path.replace('.csv', '_failed.csv')
            print(f'WARNING: {len(failed)} reaction(s) unsolved (solver error or effort limit exceeded), saved to {failed_path}; re-run with a higher work limit to retry them')
            pd.DataFrame({'rxn num': failed}).set_index('rxn num').to_csv(failed_path)

        print('All done')
        return None

    def infer_v_and_dGr(self, vi):
        ''' main func for TFBA directionality (builds a one-shot model) '''
        return self.infer_v_and_dGr_batch([vi])[0][1:]

    def infer_v_and_dGr_batch(self, vi_list, progress=None):
        ''' TFBA directionality for a list of reactions, one shared MIP model.

        The legacy per-reaction +-1e6 bounds on real_std_dGr are dropped:
        unbounded dGr objectives are detected via the UNBOUNDED solver status
        instead (numerically equivalent, mirrors utils/copt_func.py), so that
        the model can be re-used across a whole chunk. '''
        results = []
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        try:
            m = TFBA(self.GEM, thermo_constrain=self.dGr, concentration_ub=self.concentration_ub,
                     biomass_synthesis=self.biomass_synthesis, env=env,
                     work_limit=self.work_limit)
            m['model'].setParam('MIPFocus', self.MIPFocus)
            for vi in vi_list:
                try:
                    max_v = infer_v_range(m, vi, 'max')
                    min_dGr = infer_dGr_range(m, vi, 'min')
                    min_v = infer_v_range(m, vi, 'min')
                    max_dGr = infer_dGr_range(m, vi, 'max')
                    if any(x is _LIMIT_HIT for x in (max_v, min_dGr, min_v, max_dGr)):
                        pass
                    elif min_v > max_v or min_dGr > max_dGr:   # NaN passes: only definite ordering violations fail
                        pass
                    else:
                        results.append((vi, min_v, max_v, min_dGr, max_dGr))
                except Exception as e:
                    print(f'rxn {vi} TFBA failed: {e}', flush=True)
                finally:
                    _bump(progress)
        finally:
            env.dispose()

        return results

    def concurrent_infer_v_and_dGr(self, v_si=0, v_ei=None, process=16, chunk_size=200,
                                   v_list=None):
        v_ei = len(self.GEM.reactions) - 1 if v_ei is None else v_ei
        v_ei = min(v_ei, len(self.GEM.reactions) - 1)
        if v_list is None:
            v_list = range(v_si, v_ei + 1)

        if self.TFBA_res_file_path is None:
            print('Please specify the file path of TFBA result')
            return None
        elif not os.path.isfile(self.TFBA_res_file_path):
            pd.DataFrame(columns=['rxn num', 'lv', 'uv', 'ldGr', 'udGr']).to_csv(
                self.TFBA_res_file_path, index=False)

        completed = set(pd.read_csv(self.TFBA_res_file_path, index_col=0).index)
        todo = [i for i in v_list
                if (not self.GEM.reactions[i].boundary) and (i not in completed)]
        if not todo:
            print('All done')
            return None

        chunk = _chunk_size(len(todo), process, chunk_size)
        chunks = [todo[k:k + chunk] for k in range(0, len(todo), chunk)]
        counter = _MP_CTX.Value('i', 0)
        failed = []
        with _MP_CTX.Pool(process, initializer=_init_worker, initargs=(self, counter)) as pool:
            pbar = tqdm(total=len(todo), desc='Gurobi-TFBA inference')
            stop_event = threading.Event()
            pump = threading.Thread(target=_progress_pump, args=(pbar, counter, stop_event), daemon=True)
            pump.start()
            try:
                for vi_list, res in pool.imap_unordered(_run_tfba_chunk, chunks):
                    done_ids = {r[0] for r in res}
                    failed.extend(vi for vi in vi_list if vi not in done_ids)
                    _save_rows(self.TFBA_res_file_path, res, ['rxn num', 'lv', 'uv', 'ldGr', 'udGr'])
            finally:
                stop_event.set()
                pump.join()
                pbar.close()

        if failed:
            failed_path = self.TFBA_res_file_path.replace('.csv', '_failed.csv')
            print(f'WARNING: {len(failed)} reaction(s) unsolved (solver error or effort limit exceeded), saved to {failed_path}; re-run with a higher work limit to retry them')
            pd.DataFrame({'rxn num': failed}).set_index('rxn num').to_csv(failed_path)

        print('All done')
        return None

    def concurrent_optimize(self, thermo_constrain, concentration_ub, biomass_synthesis,
                            process=16, batch_size=200, obj_list=None):
        # 
        def build_model(obj):
            with gp.Env() as env:
                m = TFBA(self.GEM, thermo_constrain=thermo_constrain, concentration_ub=concentration_ub, 
                         biomass_synthesis=biomass_synthesis, env=env)
                m['model'].setParam('MIPFocus', self.MIPFocus)
                m['model'].setParam('OutputFlag', 0)
                m['model'].setParam('IntFeasTol', 1e-9)
                m['model'].setParam('FeasibilityTol', 1e-9)
                m['model'].setObjective(m['abs_error_sum'])
                m['model'].optimize()
                output = m
            return output
                
        p = Pool(process)
        res_list = []
        for i, obj in enumerate(obj_list):
            print(i, self.dGr[i])
            res = p.apply_async(func=build_model, args=(obj,))
            res_list.append([i, res])

            if ((i+1) % batch_size == 0) or (i+1 == len(obj_list)):
                p.close()
                p.join()
                
                print(f'{i//batch_size} batch done')
            
                if i+1 < len(obj_list):
                    p = Pool(process)
                
        print('All done')
        return res_list