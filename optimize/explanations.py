from cpmpy import Model

# === 1. Slack explanations ===
def get_slack_explanations(model: Model) -> list[dict]:
    expl = []
    for cons in model.constraints:
        # only binary comparisons
        if not hasattr(cons, "args") or len(cons.args) < 2:
            continue

        lhs_expr, rhs_expr = cons.args[0], cons.args[-1]

        # try to evaluate both sides numerically
        try:
            lhs_val = lhs_expr.value()
            rhs_val = rhs_expr.value()
        except Exception:
            # skip if not evaluable
            continue

        # booleans → ints
        if isinstance(lhs_val, bool): lhs_val = int(lhs_val)
        if isinstance(rhs_val, bool): rhs_val = int(rhs_val)

        slack = rhs_val - lhs_val

        expl.append({
            "constraint": str(cons),  # full text repr
            "lhs": lhs_val,
            "rhs": rhs_val,
            "slack": slack
        })

    return expl

# === 2. Reduced domain explanations ===
def get_domain_reduction(model: Model) -> list[dict]:
    """
    Reports variable domain reductions caused by propagation.
    Compares initial domains to current domains after solve.
    """
    reductions = []
    # cpmpy Model stores variables in model.vars
    for var in getattr(model, 'vars', []):
        init_low, init_up = var.MIN, var.MAX
        # After solving, var.value() gives the assigned value; domain is fixed
        try:
            cur_val = var.value()
            cur_low, cur_up = cur_val, cur_val
        except Exception:
            continue
        if (cur_low, cur_up) != (init_low, init_up):
            reductions.append({
                "variable": str(var),
                "initial_domain": (init_low, init_up),
                "final_domain": (cur_low, cur_up)
            })
    return reductions

# === 3. Conflict explanations (infeasible cores) ===
def get_conflict_explanations(model: Model) -> list[str]:
    """
    If model is infeasible, pulls conflicting constraint names.
    Requires a solver that supports conflict refiner.
    """
    try:
        core = model.refine_conflict()
        return [str(cons) for cons in core]
    except Exception:
        return []  # not supported or no conflict

# === 4. Dual/shadow price explanations ===
def get_shadow_prices(model: Model) -> list[dict]:
    """
    For linear objective problems, retrieves dual values for constraints.
    Requires an LP/dual-capable backend.
    """
    prices = []
    for cons in model.constraints:
        try:
            dual = model.solver.dual_value(cons)
            prices.append({
                "constraint": str(cons),
                "shadow_price": dual
            })
        except Exception:
            continue
    return prices

# === 5. Objective contribution explanations ===
def get_objective_contribution(model: Model) -> list[dict]:
    """
    Breaks down objective into term contributions based on solution values.
    Assumes a linear objective or skips if unavailable.
    """
    contributions = []
    # Retrieve objective expression
    obj = getattr(model, 'objective', None)
    # Skip if no objective or model.objective is a method
    if obj is None or callable(obj):
        return contributions
    # Unwrap common CPMPy objective tuple (expr, sense)
    expr = obj[0] if isinstance(obj, (tuple, list)) and len(obj) >= 1 else obj
    # Ensure expression has .args and .coeffs
    if not hasattr(expr, 'args') or not hasattr(expr, 'coeffs'):
        return contributions
    # expr.args[1] is the linear part: .args vars, .coeffs coeffs
    lin = expr.args[1]
    for var, coeff in zip(lin.args, lin.coeffs):
        # Get solved value
        try:
            val = var.value()
        except Exception:
            continue
        contributions.append({
            "variable": str(var),
            "coefficient": coeff,
            "value": val,
            "contribution": coeff * val
        })
    return contributions
