import numpy as np
import cvxopt

def solve_optimization(LfV, LgV, V, gamma, robust_clf_term=0.0, 
                       torque_A=None, torque_b=None, 
                       cbf_A=None, cbf_b=None):
    """
    Solves Unified CR-CLF-CBF-QP.
    
    Args:
        robust_clf_term (float): The value ||dV/dx||*q to subtract from CLF upper bound.
                                 (Already calculated in controller, passed here for clarity)
    """
    # 1. Setup Variables
    n_mu = 3
    n_delta = 1
    n_vars = n_mu + n_delta
    p_relaxation = 3.6 

    H = np.zeros((n_vars, n_vars))
    np.fill_diagonal(H[:n_mu, :n_mu], 2.0)
    H[n_mu, n_mu] = 2.0 * p_relaxation
    
    f = np.zeros(n_vars)

    A_list = []
    b_list = []

    # --- 2. CR-CLF Constraint ---
    # LgV*μ - δ ≤ -γV - LfV - RobustTerm
    if LfV is not None:
        clf_row = np.hstack([LgV, np.array([[-1.0]])]) 
        
        # Apply robustness margin here
        upper_bound = -gamma * V - LfV - robust_clf_term
        
        A_list.append(clf_row)
        b_list.append(np.array([[upper_bound]]))

    # --- 3. Torque Constraint ---
    if torque_A is not None and torque_b is not None:
        zeros_col = np.zeros((torque_A.shape[0], 1))
        torque_row = np.hstack([torque_A, zeros_col])
        A_list.append(torque_row)
        b_list.append(torque_b)

    # --- 4. CR-CBF Constraint ---
    # Already includes robustness penalty in 'cbf_b' from cbf_formulation.py
    if cbf_A is not None and cbf_b is not None:
        zeros_col = np.zeros((cbf_A.shape[0], 1))
        cbf_row = np.hstack([cbf_A, zeros_col])
        A_list.append(cbf_row)
        b_list.append(cbf_b)

    # 5. Solve
    if not A_list:
        return np.zeros(n_mu), True

    G_qp = np.vstack(A_list)
    h_qp = np.vstack(b_list)

    cvxopt.solvers.options['show_progress'] = False
    try:
        sol = cvxopt.solvers.qp(
            cvxopt.matrix(H), cvxopt.matrix(f), 
            cvxopt.matrix(G_qp), cvxopt.matrix(h_qp)
        )
        if sol['status'] == 'optimal':
            return np.array(sol['x']).flatten()[:n_mu], True
        else:
            return np.zeros(n_mu), False
    except ValueError:
        return np.zeros(n_mu), False