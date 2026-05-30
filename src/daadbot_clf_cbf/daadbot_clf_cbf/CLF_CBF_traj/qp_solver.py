import numpy as np
from scipy.optimize import minimize

def solve_qp(LfV, LgV, V_val, gamma, e, de, cbf_L=None, cbf_b=None):
    """
    Module 3: Optimization (Unified).
    Solves Eq (29): 
    min ||mu||^2 + p * delta^2
    s.t. 
      1. CLF: LfV + LgV*mu - delta <= -gamma*V
      2. CBF: cbf_L * mu >= cbf_b
    """
    
    p_penalty = 1000.0  # Weight for relaxation variable delta
    
    # Decision Variables: x = [mu_x, mu_y, mu_z, delta]
    # Initial guess
    x0 = np.zeros(4)

    # --- CLF Constraint ---
    # LgV*mu - delta <= -gamma*V - LfV
    # Rearranged for solver (val >= 0): 
    # (-gamma*V - LfV) - (LgV*mu - delta) >= 0
    clf_bound = -gamma * V_val - LfV
    
    def cons_clf(x):
        mu = x[:3]
        delta = x[3]
        return clf_bound - (LgV @ mu - delta)

    # --- Constraints List ---
    constraints = [{'type': 'ineq', 'fun': cons_clf}]

    # --- CBF Constraint (Optional) ---
    if cbf_L is not None:
        # cbf_L * mu >= cbf_b
        # Rearranged: cbf_L * mu - cbf_b >= 0
        def cons_cbf(x):
            mu = x[:3]
            return (cbf_L @ mu) - cbf_b
        
        constraints.append({'type': 'ineq', 'fun': cons_cbf})

    # --- Cost Function ---
    def cost_func(x):
        mu = x[:3]
        delta = x[3]
        return 0.5 * np.sum(mu**2) + p_penalty * (delta**2)
    
    # Solve
    res = minimize(cost_func, x0, constraints=constraints, method='SLSQP')
    
    if res.success:
        return res.x[:3] # Return only mu
    else:
        # Robust Fallback
        return -10.0 * e - 5.0 * de
    

    

# import numpy as np
# from scipy.optimize import minimize

# def solve_qp(LfV, LgV, V_val, gamma, e, de):
#     """
#     Module 3: Optimization.
#     Solves Eq (16): min ||mu||^2 s.t. LfV + LgV*mu <= -gamma*V
#     """
    
#     # Inequality Constraint: LfV + LgV*mu <= -gamma * V
#     # Rearranged for solver (cons >= 0): 
#     # -LgV*mu - LfV - gamma*V >= 0
#     # OR: bound - LgV*mu >= 0, where bound = -gamma*V - LfV
    
#     bound = -gamma * V_val - LfV
    
#     def constraint_func(mu):
#         return bound - (LgV @ mu)
    
#     # Objective: Minimize norm of mu
#     def cost_func(mu):
#         return 0.5 * np.sum(mu**2)
    
#     # Initial guess
#     mu0 = np.zeros(3)
    
#     cons = {'type': 'ineq', 'fun': constraint_func}
    
#     # Use SLSQP solver
#     res = minimize(cost_func, mu0, constraints=cons, method='SLSQP')
    
#     if res.success:
#         return res.x
#     else:
#         # Robust Fallback if QP fails
#         return -10.0 * e - 5.0 * de