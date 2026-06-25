"""
run_heatmap.py

Run the system identification methods (SINDy and Least Trimmed Squares variants) over a grid of
noise and outlier levels to produce coefficient heatmaps. The script loads
precomputed simulation datasets, fits libraries and estimators, and saves the
recovered coefficients for each condition.

Usage (example):
    python run_heatmap.py --case LORENZ --solver SINDY --pmult 1.0 --numSamples 100

The script expects data files named like:
    data/LORENZ_heatmap_100_data.npz

And will save results under:
    coeffs_heatmap/LORENZ_SINDY_1P_5_samples.npz

Functions:
    gen_lib(data, case): build feature library and compute derivatives.
    load_data(data, noise_level, outlier_percent): extract dataset for given
        noise/outlier levels from the loaded .npz structure.
    run_solver(...): run the requested solver and return coefficient matrix.

This file focuses on running evaluations for heatmap generation and storing
coefficients; it is not a general-purpose training utility.
"""

import numpy as np
import pysindy as ps
import lts
import argparse

def gen_lib(data, case):
    """
    Build the feature library for a given test case and compute time
    derivatives of the provided data using finite differences.

    Parameters:
        data: ndarray of shape (n_timesteps, n_states)
        case: str, one of 'SIR', 'LORENZ', 'LV' indicating the dynamical
              system to construct appropriate library and initial state.

    Returns:
        y_dot: ndarray, time derivatives of data
        D: ndarray, feature library evaluated on data
        lib: pysindy feature library instance
        t: time vector used for differentiation
        t_span: tuple, integration time span used for simulations
        initial_state: list, initial condition used in the simulations
    """
    if case == 'SIR':
        t_span = (0,100)
        functions = [
            lambda x: x,  # Identity function
            lambda x,y: x*y,  # Product function
        ]
        function_names = [
            lambda x: f'{x}',  # Identity function
            lambda x,y: f'{x}{y}',  # Product function:
        ]
        lib = ps.CustomLibrary(library_functions=functions, function_names=function_names)
        initial_state = [0.99, 0.01, 0]
    else:
        lib = ps.PolynomialLibrary(degree=3)
        if case == 'LORENZ':
            t_span = (0,20)
            initial_state = [1.0, 1.0, 1.0]
        else:
            t_span = (0,30)
            initial_state=[1.0,2.0]

    
    t = np.linspace(t_span[0], t_span[1], 1001)
    

    y_dot = ps.FiniteDifference()._differentiate(data, t=t)
    D = np.array(lib.fit_transform(data))
    return y_dot, D, lib, t, t_span, initial_state


def load_data(data, noise_level, outlier_percent):
    """Extract data and outlier indices from the loaded .npz structure.

    The .npz is expected to organize entries under keys like
    'noise_0.1' containing dict-like objects with keys 'data_5' and 'outInd_5'.
    """
    temp = data[f'noise_{noise_level:g}'].item()
    return temp[f'data_{outlier_percent:g}'], temp[f'outInd_{outlier_percent:g}']

def run_solver(y_dot, D, solver,eps, *args):
    """Run the requested solver and return coefficient estimates.

    Parameters mirror those used throughout the script. Additional positional
    arguments are solver-specific (e.g. p for SINDY-LTS, dt and lib for SR3/ESINDY).
    """
    if solver.upper() == 'SINDY':
        coeff = lts.SINDy(y_dot, D, eps=eps)
    elif solver.upper() == 'SINDY-LTS':
        p = args[0]
        coeff = lts.SINDy_LTS(y_dot, D, p = p, eps=eps)
    elif solver.upper() == 'SR3':
        outlier_level, dt, lib = args
        opt = ps.SR3(trimming_fraction=outlier_level*3)
        model = ps.SINDy(optimizer=opt, feature_library=lib)
        model.fit(data, dt)
        coeff = model.coefficients().T
    elif solver.upper() == 'ESINDY':
        dt, lib = args
        opt_esindy = ps.EnsembleOptimizer(opt=ps.STLSQ(threshold=eps),bagging=True)
        model = ps.SINDy(optimizer=opt_esindy, feature_library=lib)
        model.fit(data, dt)
        coeff = model.coefficients().T
    else:
        raise Exception('Invalid solver')
    return coeff

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run evaluation with ILTS and SINDy')
    parser.add_argument('--case', type=str, choices=['SIR', 'LORENZ','LV'], required=True,
                        help='Test case to run')
    parser.add_argument('--solver', type=str, choices=['SINDY', 'SINDY-LTS', 'SR3', 'ESINDY'], required=True,
                        help='Solver to use')
    parser.add_argument('--pmult', type=float, default=1.,
                        help='Multiplier of trusted points for ILTS')
    parser.add_argument('--numSamples', type=int, default=1,
                        help='Number of samples in the data file')
    args = parser.parse_args()


    if args.case.upper() == 'SIR':
        eps = 1e-2 
    elif args.case.upper() == 'LORENZ':
        eps = 1e-1 
    else:
        eps = 5e-2 

    noise_list = np.linspace(0,0.2,9)[1:]
    outlier_list = np.linspace(0,0.2,9)[1:]
    nSamples = args.numSamples
    simulations = np.load(f'data/{args.case.upper()}_heatmap_{nSamples}_data.npz', allow_pickle=True)
    save_dct = {}

    # Special case: noise_level and outlier_percent equal to zero
    result_dct = {}
    data, out_idx = load_data(simulations, 0, 0)
    p = int(1001 - (100*args.pmult))
    y_dot, D, lib, t, t_span, initial_state = gen_lib(data, args.case.upper())
    dt = t[1] - t[0]
    if args.solver.upper() == 'SINDY-LTS':
        input_args = [p]
    elif args.solver.upper() == 'SR3':
        input_args = [0, dt, lib]
    elif args.solver.upper() == 'ESINDY':
        input_args = [dt, lib]
    else:
        input_args = []

    coeff = run_solver(y_dot, D, args.solver, eps, *input_args)
    if args.solver.upper() == 'SINDY-LTS':
        coeff, td = coeff
        result_dct[f'outliers_0'] = td
    result_dct[f'coeffs_0'] = coeff
    save_dct[f'noise_0'] = result_dct
    for nl in noise_list:
        result_dct = {}
        for outlier_level in outlier_list:
            data_list, out_idx_list = load_data(simulations, nl, outlier_level)
            coeff_list = []
            td_list = []
            for data, out_idx in zip(data_list, out_idx_list):
                p = int(1001 - (out_idx.shape[0]*args.pmult))
                y_dot, D, lib, t, t_span, initial_state = gen_lib(data, args.case.upper())
                dt = t[1] - t[0]
                if args.solver.upper() == 'SINDY-LTS':
                    input_args = [p]
                elif args.solver.upper() == 'SR3':
                    input_args = [outlier_level, dt, lib]
                elif args.solver.upper() == 'ESINDY':
                    input_args = [dt, lib]
                else:
                    input_args = []
                
                coeff = run_solver(y_dot, D, args.solver,eps, *input_args)
                if args.solver.upper() == 'SINDY-LTS':
                    coeff, td = coeff   
                    td_list.append(td)
                coeff_list.append(coeff)
                
            if args.solver.upper() == 'SINDY-LTS':
                result_dct[f'outliers_{outlier_level*100:g}'] = np.array(td_list)
            result_dct[f'coeffs_{outlier_level*100:g}'] = np.array(coeff_list)
        save_dct[f'noise_{nl:g}'] = result_dct
    np.savez(f'coeffs_heatmap/{args.case.upper()}_{args.solver.upper()}_{args.pmult:g}P_{nSamples}_samples', allow_pickle=True, **save_dct)
