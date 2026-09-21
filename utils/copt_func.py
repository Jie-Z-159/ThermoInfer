"""

COPT (Cardinal Optimizer) based TFBA implementation.

Mirrors ``utils.func`` (the Gurobi reference implementation) using ``coptpy``,
so both solvers can coexist and be cross-validated.

Performance notes (vs. the naive one-model-per-reaction approach)
-----------------------------------------------------------------
* Each worker process builds the TFBA MIP **once** and re-uses it for a whole
  chunk of reactions (only the objective is changed between solves). This
  removes the per-reaction model build cost, which dominates on genome-scale
  models when using a Python-level modeling API.
* The tGEM object is pickled to each worker exactly once (Pool initializer)
  instead of once per task.
* The dGr bounding constraints (real_std_dGr[vi] +- 1e6) used in the Gurobi
  version are not added; unbounded dGr objectives are detected via the COPT
  UNBOUNDED status instead (numerically equivalent).

Other notes
-----------
* coptpy has no MVar matrix API, so stoichiometric expressions are built
  explicitly with ``coptpy.quicksum`` over the nonzero entries of S.
* In coptpy 8.x the 'LogToConsole' parameter does not reliably silence the
  solver (the C layer writes straight to fd 1), so every COPT call is wrapped
  in an fd-level ``_quiet`` redirect instead.
"""

from __future__ import annotations

import math
import os
import threading
import multiprocessing
import numpy as np
import pandas as pd
import cobra
import coptpy as cp
from coptpy import COPT
from tqdm import tqdm
from time import strftime, gmtime, perf_counter

from .constants import *


RT = R * default_T
ln10 = math.log(10)

_INF = cp.COPT.INFINITY

# Explicit 'fork' context: shared multiprocessing.Value counters only work
# through inheritance; Python >= 3.14 defaults to forkserver on Linux, where
# the counter would silently degrade to a per-worker copy.
_MP_CTX = multiprocessing.get_context('fork')

# Fixed solver settings, kept in sync with the Gurobi reference (utils.func):
#   RelGap = 1e-3            (Gurobi: MIPGap set per solve in infer_*_range)
#   FeasTol / IntTol = 1e-9  (Gurobi: FeasibilityTol / IntFeasTol)
#   Threads = 1: parallelism is done at the process level. MIP thread
#   scaling is inefficient, and many workers x auto threads oversubscribe
#   the machine. Scale throughput via --processes instead.
_REL_GAP = 1e-3
_FEAS_TOL = 1e-9
_THREADS_PER_WORKER = 1

# Deterministic per-solve effort cap. COPT has no work-unit limit (the Gurobi
# WorkLimit equivalent), so NodeLimit is the only machine-independent knob;
# TimeLimit is a wall-clock backstop for the one case NodeLimit cannot see
# (solves stuck at the root node, where the node count never grows). Only if
# the TimeLimit backstop triggers is the result machine-dependent. Hitting
# either limit falls through to the non-OPTIMAL branch of infer_*_range
# ('undetermined' NaN, or +/-inf for unanchored dGr).
_NODE_LIMIT = 10000
_TIME_LIMIT = 300.0

# statuses meaning "solver stopped early on an effort limit": such solves have
# no usable answer, so the reaction is routed to _failed.csv (not recorded as
# NaN) and is retried automatically if a later run uses a higher limit
_LIMIT_STATUSES = (COPT.NODELIMIT, COPT.TIMEOUT, COPT.ITERLIMIT)
_LIMIT_HIT = object()


def _quiet(fn, *args, **kwargs):
    """Call fn with fd-1 redirected to /dev/null to suppress COPT console log."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old = os.dup(1)
    try:
        os.dup2(devnull, 1)
        return fn(*args, **kwargs)
    finally:
        os.dup2(old, 1)
        os.close(old)
        os.close(devnull)


def _new_model(env, name='TFBA', gap=_REL_GAP, threads=_THREADS_PER_WORKER):
    m = env.createModel(name)
    _quiet(m.setParam, 'LogToConsole', 0)
    _quiet(m.setParam, 'RelGap', float(gap))
    _quiet(m.setParam, 'FeasTol', _FEAS_TOL)
    _quiet(m.setParam, 'IntTol', _FEAS_TOL)
    if threads is not None and threads != 0:
        _quiet(m.setParam, 'Threads', threads)
    _quiet(m.setParam, 'NodeLimit', _NODE_LIMIT)
    _quiet(m.setParam, 'TimeLimit', _TIME_LIMIT)
    return m


def _dot(coeffs, var_list):
    """Sparse dot product as a coptpy linear expression (or None if empty)."""
    nz = np.nonzero(coeffs)[0]
    if len(nz) == 0:
        return None
    return cp.quicksum(float(coeffs[k]) * var_list[k] for k in nz)


def _has_nan(values):
    return any(v is None or (isinstance(v, float) and np.isnan(v)) for v in values)


def TFBA(model: cobra.core.model.Model,
         thermo_constrain: np.ndarray = None,
         concentration_ub: float = None,
         biomass_synthesis: float = None,
         abs_v_sum: float = None,
         env=None) -> dict:
    """COPT counterpart of ``utils.func.TFBA``."""
    if env is None:
        env = _quiet(cp.Envr)
    m = _new_model(env, f'{model.id} TFBA model')

    output = {'env': env, 'model': m}

    ''' Parameters of the network '''
    n_rxn = len(model.reactions)
    n_met = len(model.metabolites)
    lv, uv = np.array([(rxn.lower_bound, rxn.upper_bound) for rxn in model.reactions]).T
    v = [_quiet(m.addVar, lb=float(lv[j]), ub=float(uv[j]), vtype=COPT.CONTINUOUS, name=f'v[{j}]')
         for j in range(n_rxn)]
    output['v'] = v

    ''' Flux balance '''
    S = cobra.util.array.create_stoichiometric_matrix(model)  # shape = [met, rxn]
    fb_cons = [_dot(S[i], v) == 0 for i in range(n_met)]
    _quiet(m.addConstrs, [c for c in fb_cons if c is not None])

    ''' Thermodynamic constraints: make sure obey the law '''
    if thermo_constrain is not None:
        if concentration_ub:
            raise NotImplementedError('concentration_ub is not supported in the COPT '
                                      'implementation (piecewise quadratic concentration cap)')

        try:
            lz, uz = np.array([(met.lz, met.uz) for met in model.metabolites]).T
        except Exception:
            lz = np.full(len(model.metabolites), -9.0)
            uz = np.full(len(model.metabolites), -1.0)
        z = [_quiet(m.addVar, lb=float(lz[i]), ub=float(uz[i]), vtype=COPT.CONTINUOUS, name=f'z[{i}]')
             for i in range(n_met)]
        output['z'] = z

        # std are added for loose the constraint
        transformed_standard_dGr_prime = thermo_constrain[:, 0]
        transformed_standard_dGr_prime_std = thermo_constrain[:, 1]
        std_dGr_uncertainty = transformed_standard_dGr_prime_std * 3 + 5

        have_dGr = (~np.isnan(transformed_standard_dGr_prime))

        std_dGf = [_quiet(m.addVar, lb=-_INF, ub=_INF, vtype=COPT.CONTINUOUS, name=f'std_dGf[{i}]')
                   for i in range(n_met)]
        output['std_dGf'] = std_dGf

        # real_std_dGr[j] = S[:, j] . std_dGf
        real_std_dGr = [_dot(S[:, j], std_dGf) for j in range(n_rxn)]
        output['real_std_dGr'] = real_std_dGr

        # real_dGr[j] = real_std_dGr[j] + RT*ln10 * (S[:, j] . z)
        real_dGr = [
            (real_std_dGr[j] if real_std_dGr[j] is not None else 0)
            + (RT * ln10) * (_dot(S[:, j], z) if _dot(S[:, j], z) is not None else 0)
            for j in range(n_rxn)]
        output['real_dGr'] = real_dGr

        # std_dGr_error = predicted - real, bounded by +- (3*SD + 5) where dGr is known
        err_cons = []
        for j in np.where(have_dGr)[0]:
            if real_std_dGr[j] is None:
                continue
            err_cons.append(real_std_dGr[j] >= transformed_standard_dGr_prime[j] - std_dGr_uncertainty[j])
            err_cons.append(real_std_dGr[j] <= transformed_standard_dGr_prime[j] + std_dGr_uncertainty[j])
        if err_cons:
            _quiet(m.addConstrs, err_cons)

        # a reaction's dGr range is bounded iff all its metabolites are anchored
        # by at least one dG-constrained reaction
        anchored_met = np.abs(S[:, have_dGr]).sum(axis=1) > 0
        col_nz = [np.nonzero(S[:, j])[0] for j in range(n_rxn)]
        anchored = np.array([len(idx) > 0 and bool(anchored_met[idx].all())
                             for idx in col_nz])
        output['anchored'] = anchored

        # add instrumental variable a0 and constant M0 to linearize the constraint
        # make sure that dGr and v have contrary signs
        # when u_dGr>=0 and l_dGr<=0, then a0=[0 or 1], v is free
        # elif l_dGr<=u_dGr<=0, then a0=1, which means v>=0
        # elif 0<=l_dGr<=u_dGr, then a0=0, which means v<=0
        a0 = [_quiet(m.addVar, lb=0.0, ub=1.0, vtype=COPT.BINARY, name=f'a0[{j}]')
              for j in range(n_rxn)]
        output['a0'] = a0

        bigM_cons = []
        for j in range(n_rxn):
            bigM_cons.append(v[j] >= -1000.0 * (1.0 - a0[j]))  # when 0<=v<=1000, a0=1
            bigM_cons.append(v[j] <= 1000.0 * a0[j])           # when -1000<=v<=0, a0=0

        for j in np.where(have_dGr)[0]:
            if real_dGr[j] is None:
                continue
            bigM_cons.append(real_dGr[j] >= -4000.0 * a0[j] + 0.0001)        # a0=0 -> v<=0, dGr>=1e-4
            bigM_cons.append(real_dGr[j] <= 4000.0 * (1.0 - a0[j]) - 0.0001)  # a0=1 -> v>=0, dGr<=-1e-4
        if bigM_cons:
            _quiet(m.addConstrs, bigM_cons)

    ''' biomass synthesis is necessary'''
    c = np.array([rxn.objective_coefficient for rxn in model.reactions])
    biomass_v = [v[j] for j in np.where(c != 0)[0]]
    output['biomass_v'] = biomass_v
    if biomass_synthesis is not None and len(biomass_v) > 0:
        _quiet(m.addConstr, cp.quicksum(biomass_v) >= biomass_synthesis)

    '''abs v sum'''
    abs_v = [_quiet(m.addVar, lb=0.0, ub=1000.0, vtype=COPT.CONTINUOUS, name=f'abs_v[{j}]')
             for j in range(n_rxn)]
    output['abs_v'] = abs_v
    abs_cons = []
    for j in range(n_rxn):
        abs_cons.append(abs_v[j] >= v[j])
        abs_cons.append(abs_v[j] >= -v[j])
    _quiet(m.addConstrs, abs_cons)
    if abs_v_sum is not None:
        _quiet(m.addConstr, cp.quicksum(abs_v[j] for j in range(n_rxn)) <= abs_v_sum)

    return output


def infer_v_range(model_dict, v_num, sense='min'):
    # main func
    m = model_dict['model']
    _quiet(m.setParam, 'RelGap', _REL_GAP)
    _quiet(m.setObjective, model_dict['v'][v_num],
           COPT.MINIMIZE if sense == 'min' else COPT.MAXIMIZE)
    _quiet(m.solve)
    status = m.status
    if status == COPT.OPTIMAL:
        val = m.objval
        return 0.0 if val == -0.0 or abs(val) < 1e-12 else np.round(val, 6)
    if status == COPT.UNBOUNDED:
        return -np.inf if sense == 'min' else np.inf
    if status in _LIMIT_STATUSES:
        return _LIMIT_HIT
    return np.nan


def infer_dGr_range(model_dict, v_num, sense='min'):
    # main func
    m = model_dict['model']
    _quiet(m.setParam, 'RelGap', _REL_GAP)
    _quiet(m.setObjective, model_dict['real_dGr'][v_num],
           COPT.MINIMIZE if sense == 'min' else COPT.MAXIMIZE)
    _quiet(m.solve)
    status = m.status
    if status == COPT.UNBOUNDED:
        return -np.inf if sense == 'min' else np.inf
    if status != COPT.OPTIMAL:
        if not model_dict['anchored'][v_num]:
            return -np.inf if sense == 'min' else np.inf
        if status in _LIMIT_STATUSES:
            return _LIMIT_HIT
        return np.nan
    val = m.objval
    r = 0.0 if val == -0.0 or abs(val) < 1e-12 else np.round(val, 6)
    r = np.inf if r >= 1e5 else r
    r = -np.inf if r <= -1e5 else r
    return r


# -----------------------------------------------------------------------
# multiprocessing worker plumbing: the tGEM instance is pickled to each
# worker exactly once; every worker then builds its model once and keeps
# re-using it for all chunks it receives.
# -----------------------------------------------------------------------

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
    """COPT-backed counterpart of ``utils.func.tGEM`` (model re-use variant)."""

    def __init__(self, GEM, dGr, concentration_ub=None, biomass_synthesis=None):
        self.GEM = GEM
        self.dGr = dGr
        self.concentration_ub = concentration_ub
        self.biomass_synthesis = biomass_synthesis
        self.node_limit = None   # None -> module default _NODE_LIMIT
        self.time_limit = None   # None -> module default _TIME_LIMIT
        self.FBA_res_file_path = None
        self.TFBA_res_file_path = None

    def max_biomass_v(self, thermo_constrain=True):
        ''' solve max biomass flux with COPT LP/MIP '''
        thermo_constrain = self.dGr if thermo_constrain else None
        t0 = perf_counter()
        env = _quiet(cp.Envr)
        try:
            m = TFBA(self.GEM, thermo_constrain=thermo_constrain,
                     concentration_ub=self.concentration_ub,
                     biomass_synthesis=None, env=env)
            objective = cp.quicksum(m['biomass_v']) if len(m['biomass_v']) else 0
            _quiet(m['model'].setObjective, objective, COPT.MAXIMIZE)
            _quiet(m['model'].solve)
            if m['model'].status != COPT.OPTIMAL:
                raise RuntimeError(f'max_biomass_v failed, status={m["model"].status}')
            val = m['model'].objval
        finally:
            env.close()

        t = strftime("%M:%S", gmtime(perf_counter() - t0))
        print(f'Max biomass v: {val}', t)
        return val

    # ------------------------------------------------------------------
    # single-reaction entry points (kept for API parity with utils.func)
    # ------------------------------------------------------------------

    def infer_v(self, vi):
        ''' main func for FBA directionality (builds a one-shot model) '''
        return self.infer_v_batch([vi])[0][1:]

    def infer_v_and_dGr(self, vi):
        ''' main func for TFBA directionality (builds a one-shot model) '''
        return self.infer_v_and_dGr_batch([vi])[0][1:]

    # ------------------------------------------------------------------
    # batch entry points: one model per call, many reactions
    # ------------------------------------------------------------------

    def infer_v_batch(self, vi_list, progress=None):
        ''' FBA directionality for a list of reactions, one shared LP model '''
        results = []
        env = _quiet(cp.Envr)
        try:
            m = TFBA(self.GEM, thermo_constrain=None, concentration_ub=self.concentration_ub,
                     biomass_synthesis=self.biomass_synthesis, env=env)
            # Apply custom limits if set
            if self.node_limit is not None:
                _quiet(m['model'].setParam, 'NodeLimit', int(self.node_limit))
            if self.time_limit is not None:
                _quiet(m['model'].setParam, 'TimeLimit', float(self.time_limit))
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
            env.close()

        return results

    def infer_v_and_dGr_batch(self, vi_list, progress=None):
        ''' TFBA directionality for a list of reactions, one shared MIP model '''
        results = []
        env = _quiet(cp.Envr)
        try:
            m = TFBA(self.GEM, thermo_constrain=self.dGr, concentration_ub=self.concentration_ub,
                     biomass_synthesis=self.biomass_synthesis, env=env)
            # Apply custom limits if set
            if self.node_limit is not None:
                _quiet(m['model'].setParam, 'NodeLimit', int(self.node_limit))
            if self.time_limit is not None:
                _quiet(m['model'].setParam, 'TimeLimit', float(self.time_limit))
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
            env.close()

        return results

    # ------------------------------------------------------------------
    # concurrent drivers: chunked dispatch + model re-use in workers
    # ------------------------------------------------------------------

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
            pbar = tqdm(total=len(todo), desc='COPT-FBA inference')
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
            print(f'WARNING: {len(failed)} reaction(s) unsolved (solver error or effort limit exceeded), saved to {failed_path}; re-run with a higher limit to retry them')
            pd.DataFrame({'rxn num': failed}).set_index('rxn num').to_csv(failed_path)

        print('All done')
        return None

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
            pbar = tqdm(total=len(todo), desc='COPT-TFBA inference')
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
            print(f'WARNING: {len(failed)} reaction(s) unsolved (solver error or effort limit exceeded), saved to {failed_path}; re-run with a higher limit to retry them')
            pd.DataFrame({'rxn num': failed}).set_index('rxn num').to_csv(failed_path)

        print('All done')
        return None
