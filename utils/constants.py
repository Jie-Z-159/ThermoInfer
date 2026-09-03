import math

R = 8.31e-3
default_T = 298.15
default_I = 0.25
default_pH = 7.0
default_pMg = 14.0
default_RT = R * default_T
RT = default_RT

default_condition = {'T':default_T,
                     'pH':default_pH,
                     'I':default_I,
                     'pMg':default_pMg}

cellular_conditions = {'c':{'pH':7.20, 'e_potential':0, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
                       'e':{'pH':7.40, 'e_potential':30 * 1e-3, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
                       'n':{'pH':7.20, 'e_potential':0, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
                       'r':{'pH':7.20, 'e_potential':0, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
                       'g':{'pH':6.35, 'e_potential':0, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
                       'l':{'pH':5.50, 'e_potential':19 * 1e-3, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
                       'm':{'pH':8.00, 'e_potential':-155 * 1e-3, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
                       'i':{'pH':8.00, 'e_potential':-155 * 1e-3, 'T':default_T, 'I':default_I, 'pMg':default_pMg},
                       'x':{'pH':7.00, 'e_potential':12 * 1e-3, 'T':default_T, 'I':default_I, 'pMg':default_pMg}}


standard_formation_dg_Mg = -455.3
standard_formation_dh_Mg = -467.0

default_u_concentration = 1e-1
default_l_concentration = 1e-9
default_uz = math.log10(default_u_concentration)
default_lz = math.log10(default_l_concentration)


H2O_u_concentration = 1000/18
H2O_l_concentration = 1000/18*0.9
H2O_uz = math.log10(H2O_u_concentration)
H2O_lz = math.log10(H2O_l_concentration)

cell_compartments = {'c': 'cytosol',
                     'e': 'extracellular space',
                     'n': 'nucleus',
                     'r': 'endoplasmic reticulum',
                     'g': 'golgi apparatus',
                     'l': 'lysosome',
                     'm': 'mitochondria',
                     'i': 'inner mitochondrial compartment',
                     'x': 'peroxisome/glyoxysome'}