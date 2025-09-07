# src/app/app.py
import datetime
from typing import Dict

import pandas as pd
import streamlit as st

# OR tools
import pyomo.environ as pyo

st.set_page_config(page_title="Herbicide Rotation Optimizer", layout="centered")
st.title("🌱 Herbicide Rotation Optimizer (MVP)")

# --- Load inputs ---
try:
    features = pd.read_csv("data_work/county_year_features.csv")
    programs = pd.read_csv("data_work/programs.csv")  # program_id, (name), (soas), cost_per_acre, ...
    risk_tbl = pd.read_csv("data_work/risk_by_program.csv")  # county_fips, year, program_id, R_t_m, cost_per_acre
except Exception as e:
    st.error(
        "Missing input files. Run these tasks in VS Code:\n"
        "1) Fetch CDL (Shelby County next 3 yrs)\n"
        "2) Pipeline: Train → Score → Optimize"
    )
    st.stop()

# ---- Normalize/repair programs.csv columns so 'name' and 'soas' always exist ----
_programs_cols = {c: c.strip() for c in programs.columns}
programs.rename(columns=_programs_cols, inplace=True)

if "program_id" not in programs.columns and programs.index.name == "program_id":
    programs = programs.reset_index()

if "name" not in programs.columns:
    programs["name"] = programs.get("program_id", "")
if "soas" not in programs.columns:
    programs["soas"] = ""



# Ensure minimal columns exist in programs for display even if CSV lacks them
if "program_id" not in programs.columns and programs.index.name == "program_id":
    programs = programs.reset_index()

if "name" not in programs.columns:
    programs["name"] = programs.get("program_id", "")
if "soas" not in programs.columns:
    programs["soas"] = ""

# --- Filter to Shelby County (FIPS 29205) ---
if "county_fips" in risk_tbl.columns:
    risk_tbl["county_fips"] = risk_tbl["county_fips"].astype(str)
    risk_tbl = risk_tbl[risk_tbl["county_fips"] == "29205"].copy()

# --- Pick the 3-year horizon: prefer next three years starting today ---
this_year = datetime.date.today().year
all_years = sorted(int(y) for y in risk_tbl["year"].unique())
future_years = [y for y in all_years if y >= this_year]
years = future_years[:3] if len(future_years) >= 3 else all_years[-3:]
if not years:
    st.error("No years found in risk_by_program.csv for Shelby County.")
    st.stop()

# Scope risk table to chosen years (we’ll still show all programs)
risk_view = risk_tbl[risk_tbl["year"].isin(years)].copy()

# --- Sidebar controls ---------------------------------------------------------
st.sidebar.header("⚙️ Optimization Settings")

lam = st.sidebar.slider(
    "Risk penalty weight (λ)",
    min_value=0.0, max_value=100.0, value=20.0, step=1.0,
    help="Higher = penalize resistance risk more relative to cost."
)

# Set per-year budgets
st.sidebar.subheader("Annual budgets ($/acre)")
budgets: Dict[int, float] = {}
for y in years:
    budgets[y] = st.sidebar.number_input(
        f"Budget {y}",
        min_value=0.0, max_value=500.0, value=40.0, step=1.0,
        help="Max spend per acre for that year."
    )

# --- Editable program costs (with full names) ---
st.sidebar.subheader("Program costs (editable)")
program_ids_in_horizon = sorted(risk_view["program_id"].unique())

_prog = programs.copy()
if "program_id" not in _prog.columns and _prog.index.name == "program_id":
    _prog = _prog.reset_index()

# Build a safe set of columns for the editor
editor_cols = ["program_id"]
if "name" in _prog.columns: editor_cols.append("name")
if "soas" in _prog.columns: editor_cols.append("soas")
if "cost_per_acre" in _prog.columns: editor_cols.append("cost_per_acre")

prog_edit = (
    _prog[_prog["program_id"].isin(program_ids_in_horizon)]
    .loc[:, editor_cols]
    .sort_values("program_id")
)

# Only allow editing cost
disabled_cols = [c for c in ["program_id", "name", "soas"] if c in prog_edit.columns]
prog_edit = st.sidebar.data_editor(
    prog_edit,
    use_container_width=True,
    num_rows="fixed",
    hide_index=True,
    disabled=disabled_cols,
    key="program_editor",
)

# Apply edited costs back into the risk table for the selected years
if "cost_per_acre" in prog_edit.columns:
    risk_view = (
        risk_view.drop(columns=["cost_per_acre"], errors="ignore")
                 .merge(prog_edit[["program_id", "cost_per_acre"]], on="program_id", how="left")
    )

st.caption(f"County: Shelby (29205) • Horizon: {years[0]}–{years[-1]} • Programs: {len(program_ids_in_horizon)}")

with st.expander("Show risk table used for optimization"):
    st.dataframe(risk_view.sort_values(["year", "program_id"]).reset_index(drop=True), use_container_width=True)

# --- Optimization (Pyomo + GLPK) ---------------------------------------------
def solve_schedule(
    risk_df: pd.DataFrame,
    budgets: Dict[int, float],
    lam: float
) -> pd.DataFrame:
    """
    risk_df: columns [year, program_id, R_t_m, cost_per_acre]
    budgets: dict year->cap
    lam: risk penalty weight
    """
    # Verify required cols
    req = {"year", "program_id", "R_t_m", "cost_per_acre"}
    missing = req - set(risk_df.columns)
    if missing:
        raise ValueError(f"risk_df is missing required columns: {missing}")

    years_sorted = sorted(int(y) for y in risk_df["year"].unique())
    progs_sorted = sorted(risk_df["program_id"].unique())

    # ----- Map program -> SOA from the 'programs' table already loaded above -----
    _prog_df = programs
    if "program_id" not in _prog_df.columns and _prog_df.index.name == "program_id":
        _prog_df = _prog_df.reset_index()
    if "soas" not in _prog_df.columns:
        _prog_df["soas"] = ""
    soa_map = _prog_df.set_index("program_id")["soas"].to_dict()
    all_soas = sorted({soa_map.get(k, "") for k in progs_sorted if pd.notna(soa_map.get(k, ""))})

    # Pyomo model
    model = pyo.ConcreteModel()
    model.T = pyo.Set(initialize=years_sorted, ordered=True)
    model.M = pyo.Set(initialize=progs_sorted, ordered=False)

    # Parameters (risk & cost)
    R = {(int(r.year), r.program_id): float(r.R_t_m) for r in risk_df.itertuples()}
    C = {(int(r.year), r.program_id): float(r.cost_per_acre) for r in risk_df.itertuples()}
    model.R = pyo.Param(model.T, model.M, initialize=R, default=1.0, within=pyo.NonNegativeReals, mutable=False)
    model.C = pyo.Param(model.T, model.M, initialize=C, default=0.0, within=pyo.NonNegativeReals, mutable=False)

    # Decision: x[t,m] = 1 if choose program m in year t
    model.x = pyo.Var(model.T, model.M, within=pyo.Binary)

    # Objective: minimize λ * risk + cost
    def obj_rule(m):
        return lam * sum(m.R[t, k] * m.x[t, k] for t in m.T for k in m.M) + \
               sum(m.C[t, k] * m.x[t, k] for t in m.T for k in m.M)
    model.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Exactly 1 program per year
    def one_per_year_rule(m, t):
        return sum(m.x[t, k] for k in m.M) == 1
    model.OnePerYear = pyo.Constraint(model.T, rule=one_per_year_rule)

    # Annual budget constraints
    def budget_rule(m, t):
        cap = float(budgets.get(int(t), 1e9))
        return sum(m.C[t, k] * m.x[t, k] for k in m.M) <= cap
    model.Budget = pyo.Constraint(model.T, rule=budget_rule)

    # -------- Rotation constraint: forbid same SOA in adjacent years --------
    for i in range(len(years_sorted) - 1):
        t1, t2 = years_sorted[i], years_sorted[i + 1]
        for s in all_soas:
            if not s:
                continue
            model.add_component(
                f"NoAdj_{s}_{t1}_{t2}",
                pyo.Constraint(
                    expr=sum(model.x[t1, k] for k in model.M if soa_map.get(k, "") == s) +
                         sum(model.x[t2, k] for k in model.M if soa_map.get(k, "") == s)
                         <= 1
                )
            )

    # Solve with GLPK
    solver = pyo.SolverFactory("glpk")
    if not solver.available(False):
        raise RuntimeError("GLPK solver not available. On macOS: `brew install glpk`.")
    solver.solve(model, tee=False)

    # Extract chosen programs
    rows = []
    for t in years_sorted:
        for k in progs_sorted:
            if pyo.value(model.x[t, k]) > 0.5:
                rows.append({
                    "year": int(t),
                    "program_id": k,
                    "soas": soa_map.get(k, "")
                })
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

# --- Run optimization and show results ---
if st.button("🚀 Optimize with current settings"):
    try:
        # Solve using current horizon risk table and edited costs
        sched = solve_schedule(risk_view.copy(), budgets, lam)

        # Build a robust lookup: start from full programs (authoritative list)
        horizon_ids = sorted(risk_view["program_id"].unique())
        base_lookup = programs.copy()
        if "program_id" not in base_lookup.columns and base_lookup.index.name == "program_id":
            base_lookup = base_lookup.reset_index()
        base_lookup = base_lookup[base_lookup["program_id"].isin(horizon_ids)].copy()

        # Ensure columns exist
        if "name" not in base_lookup.columns:
            base_lookup["name"] = base_lookup["program_id"]
        if "soas" not in base_lookup.columns:
            base_lookup["soas"] = ""

        # Bring in edited costs from the sidebar editor (if present)
        if "cost_per_acre" in prog_edit.columns:
            base_lookup = base_lookup.drop(columns=["cost_per_acre"], errors="ignore").merge(
                prog_edit[["program_id", "cost_per_acre"]],
                on="program_id",
                how="left"
            )

        lookup = base_lookup[["program_id", "name", "soas", "cost_per_acre"]].copy()

        # Merge details onto schedule
        sched = sched.merge(lookup, on="program_id", how="left")

        # Compute per-year cost and horizon summaries
        sched = sched.sort_values("year").reset_index(drop=True)
        sched["annual_cost_per_acre"] = sched["cost_per_acre"]
        total_cost = float(sched["annual_cost_per_acre"].sum())
        avg_cost = float(sched["annual_cost_per_acre"].mean())
        years_list = list(sched["year"].astype(int))

        st.success("Optimization complete.")
        st.subheader("Optimized 3-Year Schedule")
        cols = [c for c in ["year", "program_id", "name", "soas", "annual_cost_per_acre"] if c in sched.columns]
        st.dataframe(sched.loc[:, cols], use_container_width=True)

        # Budget compliance row-by-row
        st.markdown("**Budget check by year**")
        budget_rows = []
        for _, row in sched.iterrows():
            y = int(row["year"])
            cost = float(row["annual_cost_per_acre"])
            cap = float(budgets.get(y, 1e9))
            ok = cost <= cap
            budget_rows.append({"year": y, "cost_per_acre": cost, "budget_cap": cap, "within_budget": "✅" if ok else "❌"})
        st.dataframe(
            pd.DataFrame(budget_rows).sort_values("year").reset_index(drop=True),
            use_container_width=True
        )

        # Cost summary
        st.subheader("💵 3-Year Cost Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Years", f"{years_list[0]}–{years_list[-1]}")
        col2.metric("Total $/ac (3 yr)", f"{total_cost:,.2f}")
        col3.metric("Avg $/ac / yr", f"{avg_cost:,.2f}")

        # Save & download with names included
        try:
            import os
            os.makedirs("outputs", exist_ok=True)
            sched.to_csv("outputs/rotation_schedule.csv", index=False)
        except Exception:
            pass

        st.download_button(
            label="⬇️ Download schedule CSV (with names)",
            data=sched.to_csv(index=False),
            file_name="rotation_schedule.csv",
            mime="text/csv",
        )

    except Exception as err:
        st.error(f"Optimization failed: {err}")

# Show last saved schedule (if any)
st.subheader("Last saved schedule (if any)")
try:
    saved = pd.read_csv("outputs/rotation_schedule.csv")
    # Attach names if missing
    if "name" not in saved.columns or "soas" not in saved.columns:
        base_lookup = programs.copy()
        if "program_id" not in base_lookup.columns and base_lookup.index.name == "program_id":
            base_lookup = base_lookup.reset_index()
        if "name" not in base_lookup.columns:
            base_lookup["name"] = base_lookup["program_id"]
        if "soas" not in base_lookup.columns:
            base_lookup["soas"] = ""
        saved = saved.merge(
            base_lookup[["program_id", "name", "soas"]],
            on="program_id", how="left"
        )
    saved = saved[saved["year"].isin(years)]
    st.dataframe(
        saved.loc[:, ["year", "program_id", "name", "soas", "cost_per_acre"]] if "cost_per_acre" in saved.columns
        else saved.loc[:, ["year", "program_id", "name", "soas"]],
        use_container_width=True
    )
except Exception:
    st.caption("No saved schedule yet.")