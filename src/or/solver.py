# src/or/solver.py
from typing import Dict, Iterable, Optional, Literal
import pandas as pd
import pyomo.environ as pyo


def solve_rotation(
    risk_df: pd.DataFrame,
    programs_df: pd.DataFrame,
    *,
    lam: float = 20.0,
    budgets: Dict[int, float],
    objective: Literal["min_cost_plus_risk", "max_profit_minus_risk"] = "min_cost_plus_risk",
    base_yield_bu_ac: float = 60.0,
    price_per_bu: float = 12.0,
    disallow_adjacent_same_soa: bool = True,
    limit_once_per_soa: bool = False,
    min_distinct_soas: Optional[int] = None,
    must_include: Optional[Iterable[str]] = None,   # program_ids that must appear at least once
    must_exclude: Optional[Iterable[str]] = None,   # program_ids to forbid
) -> pd.DataFrame:
    """
    Inputs
    ------
    risk_df: columns must include [year, program_id, R_t_m] and optionally cost_per_acre
    programs_df: columns must include [program_id, soas] and optionally cost_per_acre
    budgets: dict mapping year -> max $/acre that can be spent that year

    Returns
    -------
    DataFrame with columns [year, program_id, soas]
    """
    # Defensive copies
    risk_df = risk_df.copy()
    programs_df = programs_df.copy()

    # Ensure required columns exist
    for col in ["year", "program_id", "R_t_m"]:
        if col not in risk_df.columns:
            raise ValueError(f"risk_df missing required column: {col}")
    for col in ["program_id", "soas"]:
        if col not in programs_df.columns:
            # handle case where program_id is the index
            if col == "program_id" and programs_df.index.name == "program_id":
                programs_df = programs_df.reset_index()
            else:
                raise ValueError(f"programs_df missing required column: {col}")

    # Optionally exclude programs
    if must_exclude:
        risk_df = risk_df[~risk_df["program_id"].isin(set(must_exclude))]

    # Sets
    years_sorted = sorted(int(y) for y in risk_df["year"].unique())
    progs_sorted = sorted(risk_df["program_id"].unique())

    # Map program -> SOA
    soa_map = programs_df.set_index("program_id")["soas"].to_dict()
    all_soas = sorted({soa_map.get(k, "") for k in progs_sorted if pd.notna(soa_map.get(k, ""))})

    # Costs: prefer per-(year,program) from risk_df if present, else fall back to programs_df
    if "cost_per_acre" in risk_df.columns and not risk_df["cost_per_acre"].isna().all():
        C_pairs = {(int(r.year), r.program_id): float(r.cost_per_acre) for r in risk_df.itertuples()}
    else:
        c_map = programs_df.set_index("program_id")["cost_per_acre"].to_dict() if "cost_per_acre" in programs_df.columns else {}
        C_pairs = {(t, k): float(c_map.get(k, 0.0)) for t in years_sorted for k in progs_sorted}

    # Risk parameter
    R_pairs = {(int(r.year), r.program_id): float(r.R_t_m) for r in risk_df.itertuples()}

    # Build model
    m = pyo.ConcreteModel()
    m.T = pyo.Set(initialize=years_sorted, ordered=True)
    m.M = pyo.Set(initialize=progs_sorted)

    m.R = pyo.Param(m.T, m.M, initialize=R_pairs, default=1.0, within=pyo.NonNegativeReals)
    m.C = pyo.Param(m.T, m.M, initialize=C_pairs, default=0.0, within=pyo.NonNegativeReals)

    m.x = pyo.Var(m.T, m.M, within=pyo.Binary)

    # Objective
    if objective == "min_cost_plus_risk":
        expr = sum(lam * m.R[t, k] * m.x[t, k] for t in m.T for k in m.M) + \
               sum(m.C[t, k] * m.x[t, k] for t in m.T for k in m.M)
        m.OBJ = pyo.Objective(expr=expr, sense=pyo.minimize)
    else:  # "max_profit_minus_risk"
        revenue = sum((base_yield_bu_ac * price_per_bu) * m.x[t, k] for t in m.T for k in m.M)
        cost = sum(m.C[t, k] * m.x[t, k] for t in m.T for k in m.M)
        risk_pen = sum(lam * m.R[t, k] * m.x[t, k] for t in m.T for k in m.M)
        m.OBJ = pyo.Objective(expr=revenue - cost - risk_pen, sense=pyo.maximize)

    # One program per year
    def one_per_year_rule(m, t):
        return sum(m.x[t, k] for k in m.M) == 1
    m.OnePerYear = pyo.Constraint(m.T, rule=one_per_year_rule)

    # Annual budgets
    def budget_rule(m, t):
        cap = float(budgets.get(int(t), 1e9))
        return sum(m.C[t, k] * m.x[t, k] for k in m.M) <= cap
    m.Budget = pyo.Constraint(m.T, rule=budget_rule)

    # Rotation: forbid same SOA in adjacent years
    if disallow_adjacent_same_soa:
        for i in range(len(years_sorted) - 1):
            t1, t2 = years_sorted[i], years_sorted[i + 1]
            for s in all_soas:
                if not s:
                    continue
                m.add_component(
                    f"NoAdj_{s}_{t1}_{t2}",
                    pyo.Constraint(
                        expr=sum(m.x[t1, k] for k in m.M if soa_map.get(k, "") == s) +
                             sum(m.x[t2, k] for k in m.M if soa_map.get(k, "") == s) <= 1
                    )
                )

    # Stronger: each SOA at most once across the horizon
    if limit_once_per_soa:
        for s in all_soas:
            if not s:
                continue
            m.add_component(
                f"Once_{s}",
                pyo.Constraint(
                    expr=sum(m.x[t, k] for t in m.T for k in m.M if soa_map.get(k, "") == s) <= 1
                )
            )

    # Minimum SOA diversity (e.g., require >= 2 distinct SOAs in the 3-year plan)
    if min_distinct_soas and min_distinct_soas > 1:
        m.S = pyo.Set(initialize=[s for s in all_soas if s])
        m.z = pyo.Var(m.S, within=pyo.Binary)  # 1 if SOA s used at least once
        for s in m.S:
            m.add_component(
                f"UseFlag_{s}",
                pyo.Constraint(
                    expr=sum(m.x[t, k] for t in m.T for k in m.M if soa_map.get(k, "") == s) - m.z[s] >= 0
                )
            )
        m.Diversity = pyo.Constraint(expr=sum(m.z[s] for s in m.S) >= int(min_distinct_soas))

    # Must-include programs (at least once)
    if must_include:
        must_set = set(must_include)
        for pid in must_set:
            if pid in progs_sorted:
                m.add_component(
                    f"Must_{pid}",
                    pyo.Constraint(expr=sum(m.x[t, pid] for t in m.T if pid in m.M) >= 1)
                )

    # Solve
    solver = pyo.SolverFactory("glpk")
    if not solver.available(False):
        raise RuntimeError("GLPK solver not available. Install it (macOS): `brew install glpk`.")
    solver.solve(m, tee=False)

    # Extract solution
    rows = []
    for t in years_sorted:
        for k in progs_sorted:
            if (t, k) in m.x and pyo.value(m.x[t, k]) > 0.5:
                rows.append({"year": int(t), "program_id": k, "soas": soa_map.get(k, "")})

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)