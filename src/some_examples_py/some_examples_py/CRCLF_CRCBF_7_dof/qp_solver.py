import numpy as np
import cvxopt

def solve_optimization(LfV, LgV, V, gamma, robust_clf_term=0.0, torque_A=None, torque_b=None, cbf_A=None, cbf_b=None):
    """
    Conformally Robust (CR) CLF-CBF-QP Solver.
    
    Optimization Variables: x = [mu, delta]
      - mu: Control input (End-effector acceleration). Dimension detected automatically (2 or 3).
      - delta: Relaxation slack variable (Scalar).
      
    Objective:
        min  0.5 * muᵀmu + 0.5 * p * delta²
        
    Constraints:
    1. CR-CLF: LgV*mu - delta <= -gamma*V - LfV - robust_clf_term
    2. Torque: torque_A*mu <= torque_b
    3. CR-CBF: cbf_A*mu <= cbf_b (Robustness included inside b_cbf via CR-CBF formulation)
    """
    
    # ---------------------------------------------------------
    # 1. Automatic Dimension Detection
    # ---------------------------------------------------------
    u_dim = LgV.shape[1] 
    num_vars = u_dim + 1
    
    # ---------------------------------------------------------
    # 2. Setup Cost Function (P, q)
    # ---------------------------------------------------------
    slack_penalty = 1.8  # Penalty for relaxing CLF constraint (Paper Eq 29)
    P_diag = np.ones(num_vars)
    P_diag[-1] = slack_penalty
    
    P = cvxopt.matrix(np.diag(P_diag))
    q = cvxopt.matrix(np.zeros(num_vars))

    # ---------------------------------------------------------
    # 3. Build Constraints (Gx <= h)
    # ---------------------------------------------------------
    G_list = []
    h_list = []

    # --- A. CR-CLF Constraint (Robust Tracking) ---
    # Mathematical Bound: LgV*mu - delta <= -gamma*V - LfV - robust_term
    clf_row = np.zeros((1, num_vars))
    clf_row[0, :u_dim] = LgV
    clf_row[0, -1] = -1.0
    
    G_list.append(clf_row)
    # [CR UPDATE]: Added the robust_clf_term to the RHS of the CLF constraint
    clf_bound = -gamma * V - LfV - robust_clf_term
    h_list.append(np.array([[clf_bound]]))

    # --- B. Torque Constraints (Input Limits) ---
    if torque_A is not None and torque_b is not None:
        if torque_A.shape[1] != u_dim:
            raise ValueError(f"Torque Matrix dim {torque_A.shape[1]} does not match LgV dim {u_dim}")

        tau_rows = np.hstack([torque_A, np.zeros((torque_A.shape[0], 1))])
        G_list.append(tau_rows)
        h_list.append(torque_b)

    # --- C. CBF Constraints (Safety) ---
    if cbf_A is not None and cbf_b is not None:
        cbf_rows = np.hstack([cbf_A, np.zeros((cbf_A.shape[0], 1))])
        G_list.append(cbf_rows)
        h_list.append(cbf_b)

    # ---------------------------------------------------------
    # 4. Solve using CVXOPT
    # ---------------------------------------------------------
    if not G_list:
        return np.zeros(u_dim), True

    G_np = np.vstack(G_list)
    h_np = np.vstack(h_list)
    
    G_cvx = cvxopt.matrix(G_np)
    h_cvx = cvxopt.matrix(h_np)
    
    cvxopt.solvers.options['show_progress'] = False
    
    try:
        sol = cvxopt.solvers.qp(P, q, G_cvx, h_cvx)
        
        if sol['status'] == 'optimal':
            res = np.array(sol['x']).flatten()
            return res[:u_dim], True
        else:
            return np.zeros(u_dim), False
            
    except ValueError as e:
        print(f"[QP Solver Error]: {e}")
        return np.zeros(u_dim), False