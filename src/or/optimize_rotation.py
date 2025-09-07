import pandas as pd
import pyomo.environ as pyo
import datetime
import pandas as pd
import pyomo.environ as pyo

# Load risk table
risk = pd.read_csv("data_work/risk_by_program.csv")
programs = pd.read_csv("data_work/programs.csv").set_index("program_id")

# Focus on Shelby County, MO (FIPS 29205)
risk["county_fips"] = risk["county_fips"].astype(str)
risk = risk[risk["county_fips"] == "29205"].copy()

# Choose the horizon = next three years if available; else fall back to latest three
this_year = datetime.date.today().year
years_all = sorted(int(y) for y in risk["year"].unique())
future_years = [y for y in years_all if y >= this_year]
years = future_years[:3] if len(future_years) >= 3 else years_all[-3:]

# Filter the table down to those three years
risk = risk[risk["year"].isin(years)].copy()
county = risk["county_fips"].unique()[0]
df = risk[risk["county_fips"]==county].copy()

Tvals = sorted(df["year"].unique())[:3]
M = sorted(df["program_id"].unique())

base_yield = 60.0
price      = 12.0
lam        = 20.0
budget_t   = {Tvals[0]: 40.0, Tvals[1]: 40.0, Tvals[2]: 40.0}

soas = programs["soas"].to_dict()

m = pyo.ConcreteModel()
m.T = pyo.Set(initialize=range(len(Tvals)))
m.M = pyo.Set(initialize=M)
m.x = pyo.Var(m.T, m.M, within=pyo.Binary)

def R(t, prog):
    year = Tvals[t]
    return float(df[(df["year"]==year) & (df["program_id"]==prog)]["R_t_m"].iloc[0])
def C(prog):
    return float(programs.loc[prog, "cost_per_acre"])

def obj_rule(m):
    profit = sum( (base_yield*price - C(p)) * m.x[t,p] for t in m.T for p in m.M )
    risk   = sum( lam * R(t,p) * m.x[t,p] for t in m.T for p in m.M )
    return profit - risk
m.obj = pyo.Objective(rule=obj_rule, sense=pyo.maximize)

def one_per_year(m, t):
    return sum(m.x[t,p] for p in m.M) == 1
m.one_program = pyo.Constraint(m.T, rule=one_per_year)

def budget_rule(m, t):
    return sum(C(p) * m.x[t,p] for p in m.M) <= budget_t[Tvals[t]]
m.budget = pyo.Constraint(m.T, rule=budget_rule)

for t in range(len(Tvals)-1):
    for p in M:
        for q in M:
            if soas[p] == soas[q]:
                setattr(m, f"norepeat_{t}_{p}_{q}",
                        pyo.Constraint(expr= m.x[t,p] + m.x[t+1,q] <= 1))

solver = pyo.SolverFactory("glpk")
res = solver.solve(m, tee=False)

schedule = []
for t in m.T:
    chosen = [p for p in m.M if pyo.value(m.x[t,p]) > 0.5][0]
    schedule.append({
        "year": Tvals[t],
        "program_id": chosen,
        "soas": soas[chosen]
    })

out = pd.DataFrame(schedule)
out.to_csv("outputs/rotation_schedule.csv", index=False)
print("Optimal schedule:\n", out)
