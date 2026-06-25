"""
generate_heatmap_data

Utility to generate noisy time-series datasets for a heatmap grid of
noise levels and outlier percentages. Intended inputs are dynamical
systems (SIR, Lorenz, Lotka-Volterra). The script solves the system,
adds Gaussian outliers according to a noise level and outlier
percentage, and saves a structured .npz file containing samples for
each grid point.

Usage (CLI):
    python gen_heatmap_data.py --case SIR --numSamples 100

Outputs:
    data/<CASE>_heatmap_<numSamples>_data.npz

This module provides:
 - getModel(case): returns model function, time vector, args and x0.
 - solve(case): integrates the model and returns data and time.
 - addNoise(data, noise_level, outlier_percent): injects outliers.

"""

import numpy as np
import argparse
from scipy.integrate import solve_ivp

def SIR(t,x, beta, gamma):
    S,I,_ = x

    dS = -beta * S *I
    dI = beta * S *I - gamma * I
    dR = gamma * I

    return  [dS, dI, dR]

def lorenz(t, x, sigma=10, rho=28, beta=8/3):
    x,y,z = x
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

def lotka_volterra(t, x, alpha=1.0, beta=0.1):
    x, y = x
    dx = alpha * x - beta * x * y
    dy = beta * x * y - 2 * alpha * y
    return [dx, dy]


def getModel(case = 'SIR'):
    """Return model function and integration parameters for a given case.

    Parameters:
        case (str): 'SIR', 'LORENZ' or 'LV'.

    Returns:
        model (callable): ODE function f(t, x, *args).
        t (ndarray): time vector for evaluation.
        t_span (tuple): (t0, tf) integration interval.
        args (tuple): parameters passed to the model.
        x0 (list): initial condition.
    """
    if case == 'SIR':
        model = SIR
        t_span = (0,100)
        args = (0.3, 0.1)
        x0 = [0.99,0.01,0.0]
    elif case == 'LORENZ':
        model = lorenz
        t_span = (0,20)
        args = (10, 28, 8/3)
        x0 = [-8., 7., 27.]
    elif case == 'LV':
        model = lotka_volterra
        t_span = (0, 30)
        args = (1.0, 0.1)
        x0 = [1.0, 2.0]
    else:
        raise ValueError(f"Unknown case: {case}")
    t = np.linspace(*t_span, 1001)
    return model, t, t_span, args, x0

def solve(case):
    """Integrate the chosen model and return state time-series.

    Returns:
        y (ndarray): shape (n_times, n_states) time-series.
        t (ndarray): time vector used for evaluation.
    """
    model, t, t_span, args, x0 = getModel(case)
    y = solve_ivp(model, t_span, x0, t_eval=t, args=args).y.T
    return y, t

def addNoise(data, noise_level, outlier_percent):
    """Inject outlier noise into columns of a time-series array.

    Parameters:
        data (ndarray): shape (n_times, n_states).
        noise_level (float): scale factor relative to RMS of each state.
        outlier_percent (float): fraction of time points to corrupt.

    Returns:
        data_noisy (ndarray): noisy data with same shape as input.
        random_vector (ndarray): boolean mask of corrupted time indices.
    """
    random_vector = np.zeros(data.shape[0], dtype=bool)
    n_outliers = int(data.shape[0] * outlier_percent)
    random_vector[:n_outliers] = True
    np.random.shuffle(random_vector)
    data_noisy = []
    for d in data.T:
        mse_d = (np.mean(d**2))**0.5  # Root Mean Square Error of the data
        nl = noise_level * mse_d
        d[random_vector] += np.random.normal(0, nl, size=n_outliers)  # Add outliers to data
        data_noisy.append(d)
    return  np.column_stack(data_noisy), random_vector


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default='SIR', choices=['SIR', 'LORENZ', 'LV'])
    parser.add_argument('--numSamples', type=int, default=100)
    parser.add_argument('-o', '--outputDir', type=str, default='../data')
    args = parser.parse_args()

    output_dir = args.outputDir
    case = args.case.upper()
    noise_list = np.linspace(0,0.2,9)[1:]
    outlier_list = np.linspace(0,0.2,9)[1:]

    data, _ = solve(case)
    save_dct = {}
    save_dct['noise_0']={
                        f'outInd_0': np.empty(0,dtype = int),
                        f'data_0': data.copy(),
                        }
    for nl in noise_list:
        result_dct = {}
        
        for op in outlier_list:
            noise_list = []
            out_list = []
            for i in range(args.numSamples):
                noisy_data, random_vec = addNoise(data.copy(), nl, op)
                out_index = np.argwhere(random_vec)
                noise_list.append(noisy_data)
                out_list.append(out_index)
            result_dct[f'outInd_{op:g}'] = np.array(out_list)
            result_dct[f'data_{op:g}'] = np.array(noise_list)

        save_dct[f'noise_{nl:g}'] = result_dct

    np.savez(f'{output_dir}/{args.case.upper()}_heatmap_{args.numSamples}_data', allow_pickle=True, **save_dct)
