"""Vehicle-to-Grid (V2G) revenue optimization module.

This module provides a clean, reusable implementation of the MILP model
using Google OR-Tools.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from ortools.linear_solver import pywraplp


def generate_default_data(
    num_hours: int = 24,
    num_evs: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate a synthetic V2G dataset compatible with the optimization model.

    Args:
        num_hours: Number of time periods in the optimization horizon.
        num_evs: Number of electric vehicles in the fleet.
        seed: Random seed used for reproducible synthetic market prices.

    Returns:
        A dictionary containing sets, parameters, and synthetic market data.
    """
    # Core sets used across the model.
    T = range(num_hours)
    EVs = range(num_evs)
    M = ["DA", "ID"]

    battery_capacity_i = {i: 50 + 10 * i for i in EVs}
    initial_energy_i = {i: 0.60 * battery_capacity_i[i] for i in EVs}
    min_energy_i = {i: 0.20 * battery_capacity_i[i] for i in EVs}
    max_energy_i = {i: battery_capacity_i[i] for i in EVs}

    max_charge_power_i = {i: 11.0 for i in EVs}
    max_discharge_power_i = {i: 11.0 for i in EVs}

    charge_efficiency = 0.93
    discharge_efficiency = 0.92
    delta_t = 1.0

    # EV availability (1 = connected to grid, 0 = unavailable).
    availability_ti = {(t, i): 1 if 8 <= t <= 17 else 0 for t in T for i in EVs}

    # Trip-related energy withdrawals to emulate mobility demand.
    trip_energy_ti = {(t, i): 0.0 for t in T for i in EVs}
    if num_evs > 0 and num_hours > 8:
        trip_energy_ti[8, 0] = 6.0
    if num_evs > 1 and num_hours > 9:
        trip_energy_ti[9, 1] = 5.0
    if num_evs > 2 and num_hours > 17:
        trip_energy_ti[17, 2] = 4.0

    grid_capacity_t = {t: 30.0 for t in T}

    # Synthetic market prices for day-ahead and intraday markets.
    np.random.seed(seed)
    buy_price_tm: dict[tuple[int, str], float] = {}
    sell_price_tm: dict[tuple[int, str], float] = {}

    for t in T:
        for m in M:
            base_price = 0.10 if m == "DA" else 0.12
            buy_price_tm[t, m] = base_price + float(np.random.uniform(0.01, 0.05))
            sell_price_tm[t, m] = buy_price_tm[t, m] + float(np.random.uniform(0.02, 0.06))

    return {
        "T": T,
        "EVs": EVs,
        "M": M,
        "battery_capacity_i": battery_capacity_i,
        "initial_energy_i": initial_energy_i,
        "min_energy_i": min_energy_i,
        "max_energy_i": max_energy_i,
        "max_charge_power_i": max_charge_power_i,
        "max_discharge_power_i": max_discharge_power_i,
        "charge_efficiency": charge_efficiency,
        "discharge_efficiency": discharge_efficiency,
        "delta_t": delta_t,
        "availability_ti": availability_ti,
        "trip_energy_ti": trip_energy_ti,
        "grid_capacity_t": grid_capacity_t,
        "buy_price_tm": buy_price_tm,
        "sell_price_tm": sell_price_tm,
    }


def create_model(data: dict[str, Any]) -> dict[str, Any]:
    """Create solver, decision variables, objective, and constraints.

    Args:
        data: Model input dictionary produced by ``generate_default_data``
            or a custom equivalent structure.

    Returns:
        A dictionary containing the solver instance, input data, and
        optimization decision variables.
    """
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        raise RuntimeError("SCIP solver was not created. Verify OR-Tools installation.")

    T = data["T"]
    EVs = data["EVs"]
    M = data["M"]

    max_charge_power_i = data["max_charge_power_i"]
    max_discharge_power_i = data["max_discharge_power_i"]
    min_energy_i = data["min_energy_i"]
    max_energy_i = data["max_energy_i"]

    # Decision variables
    P_buy_tm = {(t, m): solver.NumVar(0, solver.infinity(), f"P_buy_{t}_{m}") for t in T for m in M}
    P_sell_tm = {(t, m): solver.NumVar(0, solver.infinity(), f"P_sell_{t}_{m}") for t in T for m in M}
    P_ch_ti = {(t, i): solver.NumVar(0, max_charge_power_i[i], f"P_ch_{t}_{i}") for t in T for i in EVs}
    P_dch_ti = {(t, i): solver.NumVar(0, max_discharge_power_i[i], f"P_dch_{t}_{i}") for t in T for i in EVs}
    E_ti = {(t, i): solver.NumVar(min_energy_i[i], max_energy_i[i], f"E_{t}_{i}") for t in T for i in EVs}

    u_ch_ti = {(t, i): solver.IntVar(0, 1, f"u_ch_{t}_{i}") for t in T for i in EVs}
    u_dch_ti = {(t, i): solver.IntVar(0, 1, f"u_dch_{t}_{i}") for t in T for i in EVs}

    # Objective: maximize trading profit.
    delta_t = data["delta_t"]
    buy_price_tm = data["buy_price_tm"]
    sell_price_tm = data["sell_price_tm"]

    revenue = solver.Sum(P_sell_tm[t, m] * sell_price_tm[t, m] * delta_t for t in T for m in M)
    cost = solver.Sum(P_buy_tm[t, m] * buy_price_tm[t, m] * delta_t for t in T for m in M)
    solver.Maximize(revenue - cost)

    # Constraints: physical consistency, market consistency, and grid limits.
    initial_energy_i = data["initial_energy_i"]
    availability_ti = data["availability_ti"]
    charge_efficiency = data["charge_efficiency"]
    discharge_efficiency = data["discharge_efficiency"]
    trip_energy_ti = data["trip_energy_ti"]
    grid_capacity_t = data["grid_capacity_t"]

    for i in EVs:
        solver.Add(E_ti[0, i] == initial_energy_i[i])

    for t in T:
        for i in EVs:
            if t > 0:
                solver.Add(
                    E_ti[t, i]
                    == E_ti[t - 1, i]
                    + charge_efficiency * P_ch_ti[t, i] * delta_t
                    - (1.0 / discharge_efficiency) * P_dch_ti[t, i] * delta_t
                    - trip_energy_ti[t, i]
                )

            solver.Add(P_ch_ti[t, i] <= max_charge_power_i[i] * u_ch_ti[t, i])
            solver.Add(P_dch_ti[t, i] <= max_discharge_power_i[i] * u_dch_ti[t, i])
            solver.Add(u_ch_ti[t, i] + u_dch_ti[t, i] <= 1)

            solver.Add(P_ch_ti[t, i] <= max_charge_power_i[i] * availability_ti[t, i])
            solver.Add(P_dch_ti[t, i] <= max_discharge_power_i[i] * availability_ti[t, i])
            solver.Add(u_ch_ti[t, i] <= availability_ti[t, i])
            solver.Add(u_dch_ti[t, i] <= availability_ti[t, i])

            solver.Add(E_ti[t, i] >= min_energy_i[i])
            solver.Add(E_ti[t, i] <= max_energy_i[i])

        solver.Add(solver.Sum(P_buy_tm[t, m] for m in M) == solver.Sum(P_ch_ti[t, i] for i in EVs))
        solver.Add(solver.Sum(P_sell_tm[t, m] for m in M) == solver.Sum(P_dch_ti[t, i] for i in EVs))
        solver.Add(solver.Sum(P_buy_tm[t, m] for m in M) <= grid_capacity_t[t])
        solver.Add(solver.Sum(P_sell_tm[t, m] for m in M) <= grid_capacity_t[t])

    return {
        "solver": solver,
        "data": data,
        "variables": {
            "P_buy_tm": P_buy_tm,
            "P_sell_tm": P_sell_tm,
            "P_ch_ti": P_ch_ti,
            "P_dch_ti": P_dch_ti,
            "E_ti": E_ti,
            "u_ch_ti": u_ch_ti,
            "u_dch_ti": u_dch_ti,
        },
    }


def solve_model(model: dict[str, Any]) -> int:
    """Solve the optimization model.

    Args:
        model: Dictionary returned by ``create_model``.

    Returns:
        OR-Tools status code.
    """
    solver = model["solver"]
    return solver.Solve()


def status_to_text(status: int) -> str:
    """Convert OR-Tools solver status code to readable text.

    Args:
        status: Integer solver status from OR-Tools.

    Returns:
        Human-readable status string.
    """
    mapping = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }
    return mapping.get(status, "UNKNOWN")


def extract_results(model: dict[str, Any], status: int) -> dict[str, Any]:
    """Extract objective, economics, and selected operational summaries.

    Args:
        model: Dictionary returned by ``create_model``.
        status: Integer solver status returned by ``solve_model``.

    Returns:
        A dictionary with solver status and optimization KPIs.
    """
    solver = model["solver"]
    data = model["data"]
    vars_ = model["variables"]

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return {
            "status": status_to_text(status),
            "message": "No feasible solution available for extraction.",
        }

    T = data["T"]
    EVs = data["EVs"]
    M = data["M"]
    delta_t = data["delta_t"]

    P_buy_tm = vars_["P_buy_tm"]
    P_sell_tm = vars_["P_sell_tm"]
    E_ti = vars_["E_ti"]

    buy_price_tm = data["buy_price_tm"]
    sell_price_tm = data["sell_price_tm"]

    total_revenue = sum(P_sell_tm[t, m].solution_value() * sell_price_tm[t, m] * delta_t for t in T for m in M)
    total_cost = sum(P_buy_tm[t, m].solution_value() * buy_price_tm[t, m] * delta_t for t in T for m in M)

    final_energy_by_ev = {i: E_ti[max(T), i].solution_value() for i in EVs}

    return {
        "status": status_to_text(status),
        "objective_value": solver.Objective().Value(),
        "total_revenue": total_revenue,
        "total_cost": total_cost,
        "net_profit": total_revenue - total_cost,
        "final_energy_by_ev": final_energy_by_ev,
    }


def export_results(results: dict[str, Any]) -> Path:
    """Export optimization results to a JSON file in the results folder.

    Args:
        results: Dictionary produced by ``extract_results``.

    Returns:
        Path to the exported JSON file.
    """
    # Ensure output directory exists even in fresh clones.
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"v2g_results_{timestamp}.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return output_file


def run_pipeline() -> dict[str, Any]:
    """Run the full optimization pipeline and persist outputs.

    Returns:
        Results dictionary including exported file path.
    """
    data = generate_default_data()
    model = create_model(data)
    status = solve_model(model)
    results = extract_results(model, status)
    exported_file = export_results(results)
    results["exported_file"] = str(exported_file)
    return results


def main() -> None:
    """Command-line entry point for quick portfolio demos."""
    results = run_pipeline()

    print("V2G Optimization Run")
    print(f"Status: {results['status']}")

    if "objective_value" in results:
        print(f"Objective Value: {results['objective_value']:.4f} EUR")
        print(f"Total Revenue: {results['total_revenue']:.4f} EUR")
        print(f"Total Cost: {results['total_cost']:.4f} EUR")
        print(f"Net Profit: {results['net_profit']:.4f} EUR")
        print("Final Energy by EV:")
        for ev, val in results["final_energy_by_ev"].items():
            print(f"  EV {ev}: {val:.2f} kWh")
    else:
        print(results["message"])

    print(f"Results exported to: {results['exported_file']}")


if __name__ == "__main__":
    main()
