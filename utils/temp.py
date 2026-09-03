import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import math
from tqdm import tqdm, trange
from functools import reduce
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from ThermoInfer.utils.constants import *



def TFBA(S_df:pd.DataFrame, 
         rxn_df:pd.DataFrame,
         met_df:pd.DataFrame,
         medium:str,
         thermo_constrain:bool = True, 
         order = None,
         kcat=10, 
         km=130*1e-6,
         env=None,
         measured_flux=None,
         direcitionality=None) -> dict:
    # 
    m = gp.Model(name=f'TFBA model', env=env)
    output = {'model':m}
    m.setParam('OutputFlag', 0)
    m.setParam('IntFeasTol', 1e-9)
    m.setParam('FeasibilityTol', 1e-9)
    
    m.setParam('FuncNonlinear', 1)
    m.setParam('MIPFocus', 3)
    m.setParam('MIPGap', 1e-5)

    '''Adjust data'''
    S = S_df.to_numpy().copy() # shape = [met, rxn]
    if measured_flux is None:
        measured_flux = rxn_df[f'Vm_{medium}'].to_numpy().copy()

    ''' Parameters of the network '''
    if direcitionality:
        lv, uv = np.array([(-1, 1) if r else (0, 1) for r in rxn_df['Reversibility']]).T#*1000
    else:
        lv, uv = (-1, 1)
    v = m.addMVar(shape=S.shape[1], vtype='C', lb=lv, ub=uv, name='v')
    output['v'] = v

    ''' Flux balance '''
    m.addConstr(S[met_df['Balanced'].to_numpy()] @ v == 0, name="fb")

    ''' Thermodynamic constraints: make sure obey the law '''
    if thermo_constrain is True:
        #print('model with thermodynamic constraints')
        # Thermodynamic Constraint
        z_adj=0
        lz = np.array([-7.0 if np.isnan(lc) else np.log10(lc) for lc in met_df[f'{medium} lc']])
        uz = np.array([-1.0 if np.isnan(uc) else np.log10(uc) for uc in met_df[f'{medium} uc']])
        
        std_dGf = m.addMVar(shape=S.shape[0], vtype='C', lb=-1e4, ub=1e4, name='std_dGf')
        z = m.addMVar(shape=S.shape[0], vtype='C', lb=lz-z_adj, ub=uz-z_adj, name='z')
        output['z'] = z + z_adj

        real_dGf = std_dGf + RT * np.log(10) * (z + z_adj)
        std_dGr = S.T @ std_dGf
        real_dGr = S.T @ real_dGf
        
        a0 = m.addMVar(shape=S.shape[1], vtype='B', name='a0')
        output['a0'] = a0
        
        m.addConstr(v[:-2] >= lv[:-2] * (1-a0[:-2]), name='v0') # when 0<=v<=uv, a0=1
        m.addConstr(v[:-2] <= uv[:-2] * a0[:-2], name='v1') # when lv<=v<=0, a0=0

        m.addConstr(real_dGr[:-2] >= -4000 * a0[:-2] + 1e-3, name='therb') # when a0=0, v<=0, u_dGr>=0
        m.addConstr(real_dGr[:-2] <= 4000 * (1-a0[:-2]) - 1e-3, name='therf') # when a0=1, v>=0, l_dGr<=0

        output['std_dGf'] = std_dGf
        output['std_dGr'] = std_dGr
        output['real_dGf'] = real_dGf
        output['real_dGr'] = real_dGr

        standard_dGr_prime, standard_dGr_prime_SD = rxn_df[['standard_dGr_prime', 'standard_dGr_prime_SD']].to_numpy().T
        standard_dGr_prime = np.nan_to_num(standard_dGr_prime, nan=0.0)
        standard_dGr_prime_SD = np.nan_to_num(standard_dGr_prime_SD, nan=1000)
        
        m.addConstr(std_dGr[:-2] <= standard_dGr_prime[:-2] + standard_dGr_prime_SD[:-2])
        m.addConstr(std_dGr[:-2] >= standard_dGr_prime[:-2] - standard_dGr_prime_SD[:-2])
        
        # Metabolates' Concentration Constraint
        c = m.addMVar(shape=S.shape[0], vtype='C', lb=10**(lz-z_adj), ub=10**(uz-z_adj), name='c')
        for i, (_, row) in enumerate(met_df.iterrows()):
            if row['MetID'] in ['biomass', 'antibody']:
                pass
            else:
                m.addGenConstrLogA(c[i], z[i], 10.0,)
        c_sum = c.sum()
        output['c'] = c * 10.0**z_adj
        output['c_sum'] = c_sum
        m.addConstr(c_sum <= 0.5, name='c_sum')
        
        
        
    # Enzyme Constraint
    e = m.addMVar(shape=S.shape[1], vtype='C', lb=0, ub=np.inf, name='e')
    #m.addConstr(e.sum() <= 0.045)
    output['e'] = e

    if order == 0:
        print(f'order = {order}')
        V_max = m.addMVar(shape=S.shape[1], vtype='C', lb=0, ub=np.inf, name='V_max')
        m.addConstr(V_max == kcat * e)
        output['V_max'] = V_max

        m.addConstr(v <= V_max)
        m.addConstr(-v <= V_max)

        if thermo_constrain is True:
            vn_over_vt = m.addMVar(shape=S.shape[1], vtype='C', lb=-1, ub=1, name='vn_over_vt')
            output['vn_over_vt'] = vn_over_vt

            b1 = m.addMVar(shape=S.shape[1], vtype='B', name='b1')
            b2 = m.addMVar(shape=S.shape[1], vtype='B', name='b2')
            b3 = m.addMVar(shape=S.shape[1], vtype='B', name='b3')
            
            m.addConstr(-40000*b1 - 6*b2 + 6*b3 <= real_dGr, name='pwl1')
            m.addConstr(-6*b1 + 6*b2 + 40000*b3 >= real_dGr, name='pwl2')
            m.addConstr(b1 + b2 + b3 == 1, name='pwl3')


            m.addConstr(vn_over_vt == b1 - b2*real_dGr/6 - b3, name='pwl4')
            
            m.addConstr(v == V_max * vn_over_vt)
    
    elif order == 1:
        print(f'order = {order}')
        _real_dGr_over_RT = m.addMVar(shape=S.shape[1], vtype='C', lb=-10000, ub=10000)
        vb_over_vf = m.addMVar(shape=S.shape[1], vtype='C', lb=0, ub=1, name='vb_over_vf')
        for i in range(S.shape[1]):
            m.addGenConstrExp(_real_dGr_over_RT[i], vb_over_vf[i])
        v_net_over_v_f = 1 - vb_over_vf
        output['v_net_over_v_f'] = v_net_over_v_f

        _Q = m.addMVar(shape=S.shape[1], vtype='C', lb=0, ub=10000, name='_Q')
        m.addConstr(_Q == e * c)
        m.addConstr(_Q * kcat * v_net_over_v_f == v, )

    elif order == 'm':
        print(f'order = {order}')
        V_max = m.addMVar(shape=S.shape[1], vtype='C', lb=0, ub=np.inf, name='V_max')
        m.addConstr(V_max == kcat * e)
        output['V_max'] = V_max
    
        vn_over_vt = m.addMVar(shape=S.shape[1], vtype='C', lb=-1, ub=1, name='vn_over_vt')
        output['vn_over_vt'] = vn_over_vt

        b1 = m.addMVar(shape=S.shape[1], vtype='B', name='b1')
        b2 = m.addMVar(shape=S.shape[1], vtype='B', name='b2')
        b3 = m.addMVar(shape=S.shape[1], vtype='B', name='b3')
        
        m.addConstr(-40000*b1 - 6*b2 + 6*b3 <= real_dGr, name='pwl1')
        m.addConstr(-6*b1 + 6*b2 + 40000*b3 >= real_dGr, name='pwl2')
        m.addConstr(b1 + b2 + b3 == 1, name='pwl3')


        m.addConstr(vn_over_vt == b1 - b2*real_dGr/6 - b3, name='pwl4')
        
        m.addConstr(v == V_max * vn_over_vt)

        km_over_Q_plus_1 = m.addMVar(shape=S.shape[1], vtype='C', lb=0, ub=np.inf, name='km_over_Q')
        output['km_over_Q_plus_1'] = km_over_Q_plus_1
        for i, f_id in enumerate(rxn_df['ReactionIDs']):
            if f_id in ['Biomass', 'mAb']:
                continue
            s = S.T[i]
            _km_over_Q_plus_1_i = 1
            for met_c, met_coeff in zip(c[s<0], s[s<0]):
                assert int(met_coeff) == met_coeff
                for _ in range(-int(met_coeff)):
                    _km_over_c = m.addVar(lb=0, ub=np.inf, vtype='C')
                    _temp = m.addVar(lb=1, ub=np.inf, vtype='C')
                    b = m.addConstr(_km_over_c * met_c == km * 10.0**-z_adj)
                    m.addConstr(_temp == _km_over_Q_plus_1_i * (1 + _km_over_c))
                    _km_over_Q_plus_1_i = _temp
            m.addConstr(km_over_Q_plus_1[i] == _km_over_Q_plus_1_i, )

        m.addConstr(V_max * vn_over_vt == v * km_over_Q_plus_1, )

    '''object'''
    Flux_Diff = 0
    for v_idx, row in rxn_df.iterrows():
        flux = measured_flux[v_idx]#*1000
        if not np.isnan(flux):
            #m.addConstr(v[v_idx] >= flux*0.2, name="measuered l")
            #m.addConstr(v[v_idx] <= flux*1.8, name="measuered u")
            flux_diff = (v[v_idx] - flux)/flux # (flux*0.05+0.0028) #(v[v_idx]*(1/flux) - 1) # 
            Flux_Diff += flux_diff*flux_diff
    output['Flux_Diff'] = Flux_Diff
    '''
    dGr_Diff = 0
    if thermo_constrain is True:
        print('SD with thermodynamic constrain')
        for i, row in rxn_df.iterrows():
            standard_dGr_prime = row['standard_dGr_prime']
            standard_dGr_prime_SD = row['standard_dGr_prime_SD']
            if not np.isnan(standard_dGr_prime):
                std_dGr_diff = (std_dGr[i] - standard_dGr_prime)#/standard_dGf_prime_SD
                dGr_Diff += std_dGr_diff*std_dGr_diff
                m.addConstr((std_dGr[i] - standard_dGr_prime) <= standard_dGr_prime_SD, name='dGr_constrain')
                m.addConstr(-(std_dGr[i] - standard_dGr_prime) <= standard_dGr_prime_SD, name='dGr_constrain')
    output['dGr_Diff'] = dGr_Diff
    '''
                
    return output





def MFA_net_flux():
    # 
    SMatrix_df = pd.read_excel('Data_yonghong/SMatrix.xlsx', index_col=0)
    DataTotal_mine_df = pd.read_excel('Data_yonghong/DataTotal_mine.xlsx', index_col=None)
    DataTotal_raw_df = pd.read_excel('Data_yonghong/DataTotal_raw.xlsx', index_col=None)
    conditions = ['normal medium','low-ammonia medium','high-ammonia medium']

    #
    k = np.zeros(shape=(SMatrix_df.shape[1], 75))
    for i, x in enumerate(SMatrix_df.T.to_numpy()):
        for j, y in enumerate(SMatrix_df.T.to_numpy()[:75]):
            if np.allclose(x, y):
                k[i,j] = 1
            elif np.allclose(x, -y):
                k[i,j] = -1

    MFA_net_fluxes = {}
    for i, condition in enumerate(conditions):
        net_flux = np.matmul(DataTotal_mine_df[condition].to_numpy(), k)
        # remove v1, v22 and v26, v73
        net_flux = np.delete(net_flux, [0,21,25,72])
        MFA_net_fluxes[condition] = net_flux

    return MFA_net_fluxes



from Bio.KEGG import REST

def find_reactions_by_cid(compound_id):
    # 使用kegg_find查找与化合物ID相关的反应
    result = REST.kegg_get(compound_id).read()
    #print(result)
    reaction_ids = []
    A = False
    for line in result.split('\n'):
        if line.startswith('REACTION'):
            parts = line.split()
            reaction_ids.extend(parts[1:])
            A = True
        elif A:
            if line.startswith('   ') or line.startswith('REACTION'):
                parts = line.split()
                reaction_ids.extend(parts)
            else:
                A = False
    return reaction_ids

# 示例用法
def find_reactions_by_compound_id(compound_list):
    reaction_id = []
    for cid in compound_list:
        rid = find_reactions_by_cid(cid)
        reaction_id.append(rid)
        
    reaction_id = reduce(lambda x, y: set(x) & set(y), reaction_id)
    return (reaction_id)



def load_data():
    # 
    #SMatrix_df = pd.read_excel('Data_yonghong/SMatrix.xlsx', index_col=0)
    #MappingAtom_raw_1_df = pd.read_excel('Data_yonghong/MappingAtom_raw_1.xlsx', index_col=None)
    #DataTotal_mine_df = pd.read_excel('Data_yonghong/DataTotal_mine.xlsx', index_col=None)
    #DataTotal_raw_df = pd.read_excel('Data_yonghong/DataTotal_raw.xlsx', index_col=None)
    #ToSimulateFlux1_1_df = pd.read_excel('Data_yonghong/ToSimulateFlux1_1.xlsx', index_col=None)
    #ToSimulateFlux1_2_df = pd.read_excel('Data_yonghong/ToSimulateFlux1_2.xlsx', index_col=None)
    #ToSimulateFlux1_3_df = pd.read_excel('Data_yonghong/ToSimulateFlux1_3.xlsx', index_col=None)

    rxn_df = pd.read_csv('../MFA_data/rxn_df.csv', index_col=0)#.reset_index()
    met_df = pd.read_csv('../MFA_data/met_df.csv', index_col=0)

    S_df = pd.DataFrame(columns=rxn_df['FluxIDs'].to_list(), )
    for idx, row in rxn_df.iterrows():
        substrate = row['SubstrateIDs(atoms)'].split('+')
        for x in substrate:
            x = x.split('*')
            coeff = float(x[0]) if len(x)==2 else 1
            met_id = x[-1] 
            met_id = met_id if met_id.find('(')==-1 else met_id[:met_id.find('(')]
            if met_id not in S_df.index:
                S_df.loc[met_id] = 0.0
            S_df.loc[met_id, row['FluxIDs']] += -coeff
        
        product = row['ProductIDs(atoms)'].split('+')
        for x in product:
            x = x.split('*')
            coeff = float(x[0]) if len(x)==2 else 1
            met_id = x[-1] 
            met_id = met_id if met_id.find('(')==-1 else met_id[:met_id.find('(')]
            if met_id not in S_df.index:
                S_df.loc[met_id] = 0.0
            S_df.loc[met_id, row['FluxIDs']] += coeff
    S_df.fillna(0, inplace=True)

    met_df = met_df.set_index(keys='MetID').loc[S_df.index.to_list(),:].reset_index()

    return rxn_df, met_df, S_df



from scipy.linalg import null_space
def random_solution(S_df, met_df, rxn_df, medium='CM', sol_count=100, n=1):
    # 构造通解的函数
    def general_solution(x_p, null_space_basis, coefficients):
        """
        构造通解：x = x_p + c1 * v1 + c2 * v2 + ... + cn * vn
        其中 v1, v2, ..., vn 是零空间的基向量，c1, c2, ..., cn 是任意常数
        """
        # 将零空间基向量与系数相乘并求和
        x_h = np.sum([c * v for c, v in zip(coefficients, null_space_basis.T)], axis=0)
        return x_h + x_p
    
    # 定义矩阵 A 
    S = S_df.to_numpy().copy()
    v0 = rxn_df[f'Vm_{medium}'].to_numpy().copy()
    balanced_met_bool = met_df['Balanced'].to_numpy().copy()
    measured_v_bool = ~np.isnan(v0)
    
    A = S[balanced_met_bool][:, ~measured_v_bool]
    SD = v0 * 0.5 #+ 1e-5

    SolPool = []
    for i in trange(sol_count):
        # 定义向量 b
        v = np.random.randn(len(v0))*SD + v0

        if True:
            model_dict = TFBA(S_df, rxn_df=rxn_df, met_df=met_df, medium=medium, thermo_constrain=False, 
                            order=None, measured_flux=v, direcitionality=True)
            model = model_dict['model']
            rxn_v = model_dict['v']
            obj = model_dict['Flux_Diff'] + (rxn_v*rxn_v).sum()
            model.setObjective(obj, GRB.MINIMIZE)
            model.optimize()
            if model.Status == 2:
                x_p = rxn_v.X[~measured_v_bool]
                v_p = rxn_v.X
            else:
                continue

        else:
            # 1. 求解特解 x_p
            # 使用最小二乘法求解特解（适用于 A 不是满秩的情况）
            b = -np.matmul(S[:,measured_v_bool], v[measured_v_bool])[balanced_met_bool]
            x_p, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # 2. 求解齐次方程组的通解 x_h
        # 计算 A 的零空间
        null_space_basis = null_space(A)

        for j in range(n):
            # 生成任意常数
            coefficients = np.random.uniform(-1, 1, null_space_basis.shape[1])  # 任意常数
            x = general_solution(x_p, null_space_basis, coefficients)

            # 打印结果
            v_p[~measured_v_bool] = x
            Sv = np.matmul(S[balanced_met_bool,:],v_p)
            if np.isclose(Sv,0).all():
                SolPool.append(v_p)
            else:
                raise ValueError(f"Sv 不等于 0, ={Sv}")
    
    return np.array(SolPool)


def nearest_TFBA_solution(S_df, met_df, rxn_df, FBA_SolPool, medium='CM', n=100):
    # 
    measured_v_bool = rxn_df['Measured'].to_numpy()
    V = []
    D = []
    L = []

    for i, fba_v in enumerate(tqdm(FBA_SolPool[:n])):
        # 
        m_v = fba_v.copy()
        m_v[~measured_v_bool] = np.nan
        model_dict = TFBA(S_df, rxn_df, met_df, medium=medium, thermo_constrain=True, order=None,
                          measured_flux=m_v, direcitionality=True)
        model = model_dict['model']
        v = model_dict['v']
        model.setParam('Timelimit', 100)
        v.Start = fba_v
        v_diff = (v[measured_v_bool]/fba_v[measured_v_bool] - 1)*1000
        loss = v_diff*v_diff
        distance = (v[~measured_v_bool] - fba_v[~measured_v_bool])*(v[~measured_v_bool] - fba_v[~measured_v_bool])

        model.setObjective(distance.sum()+loss.sum(), GRB.MINIMIZE)
        model.optimize()
        if model.Status == 2:
            V.append(v.X)
            D.append(distance.sum().getValue())
            L.append(loss.sum().getValue())
        else:
            print(i, 'No solution found')
            pass
        
    return np.array(V), np.array(D), np.array(L)