import numpy as np
import cvxopt

def solve_optimization(LfV, LgV, V, gamma, torque_A=None, torque_b=None, cbf_A=None, cbf_b=None):
    """
    Universal CLF-CBF-QP Solver (Adaptive Dimension).
    
    Optimization Variables: x = [mu, delta]
      - mu: Control input (End-effector acceleration). Dimension detected automatically (2 or 3).
      - delta: Relaxation slack variable (Scalar).
      
    Objective:
        min  0.5 * muᵀmu + 0.5 * p * delta²
        
    Constraints:
    1. CLF: LgV*mu - delta <= -gamma*V - LfV
    2. Torque: torque_A*mu <= torque_b
    3. Safety: cbf_A*mu <= cbf_b
    """
    
    # ---------------------------------------------------------
    # 1. Automatic Dimension Detection
    # ---------------------------------------------------------
    # LgV shape is (1, u_dim).
    # For 2-link planar, u_dim = 2 (x, y).
    # For 3-link spatial, u_dim = 3 (x, y, z).
    u_dim = LgV.shape[1] 
    
    # Total variables = u_dim (control) + 1 (slack)
    num_vars = u_dim + 1
    
    # ---------------------------------------------------------
    # 2. Setup Cost Function (P, q)
    # ---------------------------------------------------------
    # Minimize: 0.5 * xᵀPx + qᵀx
    # P = diag([1, ..., 1, p_slack])
    
    slack_penalty = 3.6  # Very high penalty to enforce tracking
    P_diag = np.ones(num_vars)
    P_diag[-1] = slack_penalty
    
    P = cvxopt.matrix(np.diag(P_diag))
    q = cvxopt.matrix(np.zeros(num_vars))

    # ---------------------------------------------------------
    # 3. Build Constraints (Gx <= h)
    # ---------------------------------------------------------
    G_list = []
    h_list = []

    # --- A. CLF Constraint (Tracking) ---
    # LgV*mu - delta <= -gamma*V - LfV
    # Row: [LgV_1, ..., LgV_n, -1.0]
    clf_row = np.zeros((1, num_vars))
    clf_row[0, :u_dim] = LgV
    clf_row[0, -1] = -1.0
    
    G_list.append(clf_row)
    h_list.append(np.array([[-gamma * V - LfV]]))

    # --- B. Torque Constraints (Input Limits) ---
    # torque_A*mu <= torque_b
    # Pad with 0 for delta column: [torque_A, 0]
    if torque_A is not None and torque_b is not None:
        # Verify shapes match u_dim
        if torque_A.shape[1] != u_dim:
            raise ValueError(f"Torque Matrix dim {torque_A.shape[1]} does not match LgV dim {u_dim}")

        tau_rows = np.hstack([torque_A, np.zeros((torque_A.shape[0], 1))])
        G_list.append(tau_rows)
        h_list.append(torque_b)

    # --- C. CBF Constraints (Safety) ---
    # cbf_A*mu <= cbf_b
    # Pad with 0 for delta column: [cbf_A, 0]
    if cbf_A is not None and cbf_b is not None:
        # CBF is a hard constraint (usually), but here we allow it to compete via QP
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
            # Return only the control input mu (slice off delta)
            return res[:u_dim], True
        else:
            return np.zeros(u_dim), False
            
    except ValueError as e:
        print(f"[QP Solver Error]: {e}")
        return np.zeros(u_dim), False