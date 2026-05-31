import numpy as np
from scipy.optimize import minimize

def solve_qp(LfV, LgV, V_val, gamma, e, de, cbf_L=None, cbf_b=None, robust_term=0.0):
    """
    Module 3: Optimization (Unified with Conformal Robustness).
    
    Solves:
      min ||mu||^2 + p * delta^2
    s.t. 
      1. CR-CLF: LfV + LgV*mu - delta <= -gamma*V - robust_term
      2. CBF:    cbf_L * mu >= cbf_b
    """
    
    p_penalty = 1000.0
    x0 = np.zeros(4)

    # --- CR-CLF Constraint Modification ---
    # Standard: LfV + LgV*mu - delta <= -gamma*V
    # Robust:   LfV + LgV*mu - delta <= -gamma*V - robust_term [cite: 8, 10, 145]
    #
    # We subtract robust_term to make the limit "stricter" (more negative).
    # This reserves control authority to fight the uncertainty.
    
    #                                             vvvvvvvvvvvvv
    clf_bound = -gamma * V_val - LfV - robust_term
    #                                             ^^^^^^^^^^^^^
    
    def cons_clf(x):
        mu = x[:3]
        delta = x[3]
        # Constraint form for solver: val >= 0
        # Bound - Actual >= 0  -->  Bound >= Actual
        return clf_bound - (LgV @ mu - delta)

    constraints = [{'type': 'ineq', 'fun': cons_clf}]

    # --- CBF Constraint ---
    # Note: For CR-CBF, the robustness term is usually already added 
    # to 'cbf_b' inside the crcbf_formulation.py class[cite: 10, 170, 193].
    # So no extra change is needed here for CBF.
    if cbf_L is not None:
        def cons_cbf(x):
            mu = x[:3]
            return (cbf_L @ mu) - cbf_b
        
        constraints.append({'type': 'ineq', 'fun': cons_cbf})

    def cost_func(x):
        mu = x[:3]
        delta = x[3]
        return 0.5 * np.sum(mu**2) + p_penalty * (delta**2)
    
    # Solve
    res = minimize(cost_func, x0, constraints=constraints, method='SLSQP')
    
    if res.success:
        return res.x[:3]
    else:
        # Robust Fallback
        return -10.0 * e - 5.0 * de