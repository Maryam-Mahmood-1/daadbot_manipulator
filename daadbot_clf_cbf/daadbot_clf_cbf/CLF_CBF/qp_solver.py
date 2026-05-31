import numpy as np
import cvxopt

def solve_optimization(LfV, LgV, V, gamma, torque_A=None, torque_b=None, cbf_A=None, cbf_b=None):
    """
    Solves the Unified CLF-CBF-QP as defined in the Paper Eq (29).
    
    Optimization Variables: x = [μ, δ]  (Size 4: 3 control + 1 relaxation)
    
    Objective:
        min (μ, δ)  μᵀμ + p * δ²
        
    Constraints:
    1. Relaxed CLF (Tracking):
       LgV*μ - δ ≤ -γV - LfV
       
    2. Input Saturation (Torque):
       A_torq*μ ≤ b_torq
       
    3. Safety (CBF):
       A_cbf*μ ≤ b_cbf
    """
    
    # ---------------------------------------------------------
    # 1. Setup Variables
    # ---------------------------------------------------------
    n_mu = 3
    n_delta = 1
    n_vars = n_mu + n_delta
    
    # Weight 'p' for relaxation variable delta (Paper Eq 29)
    # High value ensures we only relax tracking if absolutely necessary (e.g. Safety conflict)
    p_relaxation = 3.6 

    # H Matrix: [2*I_3,  0 ]
    #           [ 0 ,  2*p ]
    H = np.zeros((n_vars, n_vars))
    np.fill_diagonal(H[:n_mu, :n_mu], 2.0)
    H[n_mu, n_mu] = 2.0 * p_relaxation
    
    f = np.zeros(n_vars)

    # ---------------------------------------------------------
    # 2. Build Constraints (Gx ≤ h)
    # ---------------------------------------------------------
    A_list = []
    b_list = []

    # --- Constraint A: Relaxed CLF (Eq 29 Line 2) ---
    # LgV*μ ≤ -γV - LfV + δ
    # Rearranged: LgV*μ - δ ≤ -γV - LfV
    # Matrix form: [LgV, -1] * [μ, δ]ᵀ ≤ ...
    if LfV is not None:
        clf_row = np.hstack([LgV, np.array([[-1.0]])]) 
        A_list.append(clf_row)
        b_list.append(np.array([[-gamma * V - LfV]]))

    # --- Constraint B: Torque Saturation (Eq 29 Line 4, 5) ---
    # A_torq*μ ≤ b_torq
    # Matrix form: [A_torq, 0] * [μ, δ]ᵀ ≤ b_torq
    if torque_A is not None and torque_b is not None:
        zeros_col = np.zeros((torque_A.shape[0], 1))
        torque_row = np.hstack([torque_A, zeros_col])
        A_list.append(torque_row)
        b_list.append(torque_b)

    # --- Constraint C: Safety CBF (Eq 29 Line 7) ---
    # A_cbf*μ ≤ b_cbf
    # Matrix form: [A_cbf, 0] * [μ, δ]ᵀ ≤ b_cbf
    if cbf_A is not None and cbf_b is not None:
        zeros_col = np.zeros((cbf_A.shape[0], 1))
        cbf_row = np.hstack([cbf_A, zeros_col])
        A_list.append(cbf_row)
        b_list.append(cbf_b)

    # 3. Stack Matrices
    if not A_list:
        return np.zeros(n_mu), True

    G_qp = np.vstack(A_list)
    h_qp = np.vstack(b_list)

    # 4. Solve using CVXOPT
    cvxopt.solvers.options['show_progress'] = False
    P_cvx = cvxopt.matrix(H)
    q_cvx = cvxopt.matrix(f)
    G_cvx = cvxopt.matrix(G_qp)
    h_cvx = cvxopt.matrix(h_qp)

    try:
        sol = cvxopt.solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
        
        if sol['status'] == 'optimal':
            x_sol = np.array(sol['x']).flatten()
            mu = x_sol[:n_mu]     # Extract μ
            delta = x_sol[n_mu]   # Extract δ
            # print(f"Delta: {delta:.4f}") # Debugging relaxation
            return mu, True
        else:
            return np.zeros(n_mu), False
            
    except ValueError:
        return np.zeros(n_mu), False
    

    


""""Without CBF constraints - Basic CLF-QP solver."""
# import numpy as np
# import cvxopt

# def solve_optimization(LfV, LgV, V, gamma, torque_A=None, torque_b=None, cbf_A=None, cbf_b=None):
#     """
#     Solves the CLF-QP (Control Lyapunov Function based Quadratic Program).
    
#     Mathematical Formulation:
#     -------------------------
#     Optimization Variable: μ (Auxiliary Control Input)
    
#     Objective:
#         min (μ)  ½ μᵀ H μ + fᵀ μ
        
#     Subject to:
    
#     1. Stability Constraint (CLF):
#        LgV(x) μ ≤ -γ V(x) - LfV(x)
       
#     2. Input Constraints (Torque Saturation):
#        τ_min ≤ M(q)J†(u_ref + μ - J̇q̇) + n(q,q̇) ≤ τ_max
#        (Rearranged into linear form A_torq μ ≤ b_torq)
       
#     3. Safety Constraints (CBF - Optional):
#        A_cbf μ ≤ b_cbf
#     """
    
#     # ---------------------------------------------------------
#     # 1. Objective Function
#     #    Minimize control deviation energy: ||μ||²
#     #    Standard Form: min ½ xᵀ P x + qᵀ x
#     # ---------------------------------------------------------
#     # 3 variables for μ (x, y, z acceleration correction)
#     n_vars = 3
    
#     # H = 2I  (Factor of 2 because standard form is 1/2 xPx)
#     H = 2.0 * np.eye(n_vars)
    
#     # f = 0
#     f = np.zeros(n_vars)

#     # ---------------------------------------------------------
#     # 2. Build Constraints (G x ≤ h)
#     # ---------------------------------------------------------
#     A_list = []
#     b_list = []

#     # --- Constraint A: RES-CLF Stability ---
#     # Inequality: LgV ⋅ μ ≤ -γ V - LfV
#     A_list.append(LgV) 
#     b_list.append(np.array([[-gamma * V - LfV]]))

#     # --- Constraint B: Torque Saturation ---
#     # Form: A_torq ⋅ μ ≤ b_torq
#     # (Matrices pre-calculated in main controller to save time here)
#     if torque_A is not None and torque_b is not None:
#         A_list.append(torque_A)
#         b_list.append(torque_b)

#     # --- Constraint C: Control Barrier Function (Safety) ---
#     # Form: A_cbf ⋅ μ ≤ b_cbf
#     if cbf_A is not None and cbf_b is not None:
#         A_list.append(cbf_A)
#         b_list.append(cbf_b)

#     # 3. Stack Matrices for Solver
#     if not A_list:
#         return np.zeros(n_vars), True

#     G_qp = np.vstack(A_list)
#     h_qp = np.vstack(b_list)

#     # 4. Solve using CVXOPT
#     # Solves: min ½ xᵀ P x + qᵀ x  s.t.  G x ≤ h
#     cvxopt.solvers.options['show_progress'] = False
#     P_cvx = cvxopt.matrix(H)
#     q_cvx = cvxopt.matrix(f)
#     G_cvx = cvxopt.matrix(G_qp)
#     h_cvx = cvxopt.matrix(h_qp)

#     try:
#         sol = cvxopt.solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
#         mu = np.array(sol['x']).flatten()
#         return mu, True
        
#     except ValueError:
#         # Infeasible (Constraints conflict, e.g., Safety vs Limits)
#         # Returns zero correction (Fall back to nominal control)
#         return np.zeros(n_vars), False