import numpy as np
from itertools import  product
from scipy.linalg import LinAlgWarning, pascal
from sklearn.linear_model import ridge_regression
from scipy.optimize import minimize, NonlinearConstraint
import warnings
from joblib import Parallel, delayed
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from scipy.integrate import solve_ivp
import pysindy as ps


def ilts(A,b, p, max_it = 100):
    m = A.shape[0]
    if isinstance(p, (float, np.floating)):
        if not 0 <= p <= 1:
            raise ValueError('A float p must satisfy 0 <= p <= 1')
        p = max(1, int(np.ceil(p * m)))
    elif not isinstance(p, (int, np.integer)) or not 1 <= p <= m:
        raise ValueError('p must be an integer in [1, m] or a float in [0, 1]')

    # Solve LS
    xr = np.linalg.lstsq(A,b,rcond=None)[0]

    # Compute residuals
    vr = A@xr - b
    fr = np.linalg.norm(vr)

    # Sort resodials and select the p best
    I_sorted = np.argsort(vr**2,axis=0).flatten()
    Ir = I_sorted[:p] #index of the best points
    Ar = A[Ir]
    br = b[Ir]

    for _ in range(max_it):
        #solve with the best results
        xp = np.linalg.lstsq(Ar,br,rcond=None)[0]
        #compute residual
        vp =  A@xp - b
        #new best indexes
        I_sorted = np.argsort(vp**2,axis=0).flatten()
        Ir = I_sorted[:p]
        fp = np.linalg.norm(vp[Ir])
        if fp >= fr:
            return xr, fr, 1, I_sorted
        
        Ar = A[Ir]
        br = b[Ir]
        fr = fp
        xr = xp

    return xr, fr, 0, I_sorted

def ilts_search(A,b, p : int|float|None =None, max_it = 100, p_min = 0.8, p_max = 1.0):
    m = A.shape[0]
    if isinstance(p, (float, np.floating)):
        if not 0 <= p <= 1:
            raise ValueError('A float p must satisfy 0 <= p <= 1')
        p = max(1, int(np.ceil(p * m)))
    elif not isinstance(p, (int, np.integer)) or not 1 <= p <= m:
        raise ValueError('p must be an integer in [1, m] or a float in [0, 1]')

    # Solve LS
    xr = np.linalg.lstsq(A,b,rcond=None)[0]

    # Compute residuals
    vr = A@xr - b

    if p is None:
        p_min = max(1, int(np.ceil(p_min * m)))
        p_max = max(1, int(np.ceil(p_max * m)))
        crit_best = 0

        p_values = np.arange(p_min, p_max +1)
        p_list = [3 * pi - 2 *m for pi in p_values if (3 * pi - 2 *m) >0]
        for p in p_list:
            xr, fr, stats, I_sorted = iter_ilst(p, vr,xr,A ,b, max_it)
            Lpp1 = 0.5 * fr**2
            if p == p_min:
                xr_best, fr_best, stats_best, I_sorted_best = (xr, fr, stats, I_sorted)
            else:
                crit = (Lpp1 - Lp)/Lp
                if crit > crit_best:
                    xr_best, fr_best, stats_best, I_sorted_best = (xr, fr, stats, I_sorted)
                    p_best = p
                    crit_best = crit
                Lp = Lpp1
        return xr_best, fr_best, stats_best, I_sorted_best, p_best
    else:
        xr, fr, stats, I_sorted = iter_ilst(p, vr,xr,A ,b, max_it)
    return  xr, fr, stats, I_sorted, p

def iter_ilst(p, vr,xr,A ,b, max_it):
    # Sort resodials and select the p best
    fr = np.linalg.norm(vr)
    I_sorted = np.argsort(vr**2,axis=0).flatten()
    Ir = I_sorted[:p] #index of the best points
    Ar = A[Ir]
    br = b[Ir]

    for _ in range(max_it):
        #solve with the best results
        xp = np.linalg.lstsq(Ar,br,rcond=None)[0]
        #compute residual
        vp =  A@xp - b
        #new best indexes
        I_sorted = np.argsort(vp**2,axis=0).flatten()
        Ir = I_sorted[:p]
        fp = np.linalg.norm(vp[Ir])
        if fp >= fr:
            return xr, fr, 1, I_sorted
        
        Ar = A[Ir]
        br = b[Ir]
        fr = fp
        xr = xp
    return xr, fr, 0, I_sorted



def SINDy_LTS(x_dot, D, p, threshold=1e-1, alpha = 0.0, max_it=2000):
    n = x_dot.shape[-1]
    m = x_dot.shape[0]
    nD = D.shape[-1]

    if isinstance(p, (float, np.floating)): # is p is a percentage
        if not 0 <= p <= 1:
            raise ValueError('A float p must satisfy 0 <= p <= 1')
        p = max(1, int(np.ceil(p * m)))

    Xi = np.zeros((nD, n))
    I_sorted = np.zeros((m,n))
    y_dot = np.zeros((p,1))
    for i in range(n):
        _, _, stats, Is = ilts(D, x_dot[:,i], p, max_it)
        I_sorted[:,i] = Is
        Ir = Is[:p]
        if stats == 0:
            print('[Warning] LOVO not finished successefuly')
        Ap = D[Ir]
        y_dot[:,0] = x_dot[Ir,i]
        Ci = SINDy(y_dot, Ap, threshold=threshold,alpha=alpha)
        Xi[:,i:i+1] = Ci
    return Xi, I_sorted


def SINDy_LTS_search(x_dot, D, p=None, threshold=1e-1, alpha = 0.0, max_it=2000, p_min = 0.8, p_max = 1.0):
    n = x_dot.shape[-1]
    m = x_dot.shape[0]
    nD = D.shape[-1]

    if isinstance(p, (float, np.floating)): # is p is a percentage
        if not 0 <= p <= 1:
            raise ValueError('A float p must satisfy 0 <= p <= 1')
        p = max(1, int(np.ceil(p * m)))

   
    Xi = np.zeros((nD, n))
    I_sorted = np.zeros((m,n))
    y_dot = np.zeros((m,1))
    ps = np.zeros(n)
    for i in range(n):
        _, _, stats, Is, pi = ilts_search(D, x_dot[:,i], p, max_it, p_min, p_max)
        I_sorted[:,i] = Is
        Ir = Is[:pi]
        if stats == 0:
            print('[Warning] LOVO not finished successefuly')
        Ap = D[Ir]
        y_dot[:pi,0] = x_dot[Ir,i]
        Ci = SINDy(y_dot[:pi], Ap, threshold=threshold,alpha=alpha)
        Xi[:,i:i+1] = Ci
        ps[i] = pi

    return Xi, I_sorted, ps

def SINDy(x_dot, D, threshold = 1e-2, alpha = 0.0, scaling_eps = False):
    n = x_dot.shape[-1]
    nD = D.shape[-1]

    # Xi = np.zeros((nD, n))
    Xi = np.linalg.lstsq(D,x_dot,rcond=None)[0]
    if scaling_eps:
        threshold = threshold / np.linalg.norm(D,axis = 0)
    for i in range(n):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=LinAlgWarning)
            Ei = ridge_regression(D,x_dot[...,i],alpha)
            assert isinstance(Ei, np.ndarray)
        nnz = np.count_nonzero(Ei)
        old_nnz = nD+1
        while nnz != old_nnz:
            small_ind = np.abs(Ei) < threshold # pyright: ignore[reportCallIssue]
            Ei[small_ind] = 0
            bi = ~small_ind

            old_nnz = nnz
            nnz = np.count_nonzero(Ei)
            if nnz == 0:
                break
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=LinAlgWarning)
                Ei[bi] = ridge_regression(D[..., bi],x_dot[...,i],alpha)

        Xi[:,i] = Ei
    return Xi
            

#Utils
def print_model(C, f_names, precision = 3):
    d = C.shape[1]

    for i in range(d):
        msg = f"x{i}' = " 
        ind = np.argwhere(~np.isclose(C[:,i],0,atol=10**(-precision))) .flatten()
        if len(ind) == 0:
            msg += '0'
        else:
            for j in ind[:-1]:
                msg += f"{C[j,i]:0.{precision}f} {f_names[j]} + "
            msg += f"{C[ind[-1],i]:0.{precision}f} {f_names[ind[-1]]} "
        print(msg)

def FUN_SINDy(t, x, solu,  lib,  z_original_part_max = 1, sel_ind =  None, u=None):
    # Assume Dict_gen_3d is a function that you've defined elsewhere
    d = x.shape[0]
    if u is not None:
        ti = int(t)
        ui = u[ti]#.reshape((1,-1))
        x = np.concatenate((x, ui), axis=0)
    x_in = (x / z_original_part_max)[None, :]
    D_temp = np.array(lib.transform(x_in))
    if sel_ind is not None:
        D_temp = D_temp[sel_ind]
    f = np.zeros(d)  # Initialize f as a 3x1 vector
    for i in range(d):
        f[i] = (z_original_part_max * D_temp @ solu[:, i:i+1]).item()
    return f

def simulate(coeff, lib, x0, tspan, t_eval, z_original_part_max = 1.,sel_ind =  None, u=None, **kwargs):
    y = solve_ivp(FUN_SINDy, tspan, x0 ,t_eval=t_eval, vectorized=False, args=(coeff, lib, z_original_part_max, sel_ind, u), **kwargs).y.T
    return y


################# WIP

def nonuni_der(X,t):
    h0 = (t[1:-1] - t[:-2])[:, None]
    h1 = (t[2:] - t[1:-1])[:, None]
    x_dot = np.zeros_like(X)
    x_dot[0] = (X[1] - X[0]) / h0[0]
    x_dot[-1] = (X[-1] - X[-2]) / h1[-1]
    x_dot[1:-1] = (h1 - h0) / (h0 * h1) * X[1:-1] + (1 / (h0 + h1)) * ((h0 / h1) * X[2:] - (h1 / h0) * X[:-2])
    return x_dot


def ilts_search_recalc(A,x,t, p : int|float|None =None, max_it = 100, p_min = 0.8, p_max = 1.0):
    m = A.shape[0]
    if isinstance(p, (float, np.floating)):
        if not 0 <= p <= 1:
            raise ValueError('A float p must satisfy 0 <= p <= 1')
        p = max(1, int(np.ceil(p * m)))
    elif not isinstance(p, (int, np.integer)) or not 1 <= p <= m:
        raise ValueError('p must be an integer in [1, m] or a float in [0, 1]')

    # Solve LS
    b = nonuni_der(x,t)
    xr = np.linalg.lstsq(A,b,rcond=None)[0]
    # Compute residuals
    vr = A@xr - b


    if p is None:
        p_min = max(1, int(np.ceil(p_min * m)))
        p_max = max(1, int(np.ceil(p_max * m)))
        crit_best = 0
        for p in range(p_min, p_max+1):
            xr, fr, stats, I_sorted = iter_ilst_recalc_b(p, vr,xr,A ,x,t, max_it)
            Lpp1 = 0.5 * fr**2
            if p == p_min:
                xr_best, fr_best, stats_best, I_sorted_best = (xr, fr, stats, I_sorted)
            else:
                crit = (Lpp1 - Lp)/Lp
                if crit > crit_best:
                    xr_best, fr_best, stats_best, I_sorted_best = (xr, fr, stats, I_sorted)
                    p_best = p
                    crit_best = crit
                Lp = Lpp1
        return xr_best, fr_best, stats_best, I_sorted_best, p_best
    else:
        xr, fr, stats, I_sorted = iter_ilst_recalc_b(p, vr,xr,A ,x,t, max_it)
    return  xr, fr, stats, I_sorted, p

def iter_ilst_recalc_b(p, vr, xr, A, x, t, max_it, max_outer=100):
    # Outer loop: alternate between (a) converging the trimmed subset under a
    # FIXED derivative target (inner C-steps, monotonic descent guaranteed,
    # same as iter_ilst), and (b) refreshing that target from the converged
    # subset. SSE isn't comparable across different targets, so convergence
    # here is judged by subset stability (the subset reproduces itself under
    # the target it induces), not by comparing residual norms across outer
    # steps.
    if max_outer < 1:
        raise ValueError('max_outer must be >= 1')

    b = nonuni_der(x, t)
    seen = set()

    for _ in range(max_outer):
        xr, fr, stats, I_sorted = iter_ilst(p, vr, xr, A, b, max_it)
        Ir = np.sort(I_sorted[:p])
        key = tuple(Ir)

        # Check against the whole history, not just the last subset: the
        # refresh map can settle into a longer cycle (A -> B -> A -> ...)
        # instead of an immediate repeat, and comparing only to the previous
        # iterate would never catch that.
        if key in seen:
            return xr, fr, -1, I_sorted
        seen.add(key)

        br = nonuni_der(x[Ir], t[Ir])
        b = nonuni_der(x, t)
        b[Ir] = br
        vr = A @ xr - b

    return xr, fr, -1, I_sorted