# -*- coding: utf-8 -*-
"""
run_heuristic.py

Runs GT+ATC heuristic for any scenario (data-only file with get_data()).

Folder layout (your current setup):
src/Heuristic/
  HeuristicBaseModel.py
  HeuristicPlots.py
  run_heuristic.py
  scenario/
    Heuristic Machine Type.py
    OtherScenario.py
    ...

Examples:
  # from src/Heuristic:
  python .\run_heuristic.py --list
  python .\run_heuristic.py --scenario "Heuristic Machine Type.py"

  # from project root:
  python .\src\Heuristic\run_heuristic.py --scenario "Heuristic Machine Type.py"
"""

import os
import argparse
import importlib.util


def import_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def list_py_files(folder: str):
    """List all .py files in a folder (for scenarios)."""
    if not os.path.isdir(folder):
        return []
    return sorted([fn for fn in os.listdir(folder) if fn.lower().endswith(".py")])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        default="Heuristic Machine Type.py",
        help='Scenario .py file under "scenario/" defining get_data()'
    )
    parser.add_argument("--k1", type=float, default=2.0, help="ATC parameter k1")
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting")
    parser.add_argument("--list", action="store_true", help='List available scenarios in "scenario/"')
    args = parser.parse_args()

    # Folder of THIS script: src/Heuristic
    here = os.path.dirname(os.path.abspath(__file__))

    # Scenario folder: src/Heuristic/scenario
    scenario_dir = os.path.join(here, "scenario")

    if args.list:
        print('\nAvailable scenario files in "scenario/":')
        files = list_py_files(scenario_dir)
        if not files:
            print(" (none found)")
        else:
            for fn in files:
                print(" -", fn)
        return

    base_path = os.path.join(here, "HeuristicBaseModel.py")
    plots_path = os.path.join(here, "HeuristicPlots.py")
    scenario_path = os.path.join(scenario_dir, args.scenario)

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base model not found: {base_path}")
    if not os.path.exists(plots_path):
        raise FileNotFoundError(f"Plots module not found: {plots_path}")
    if not os.path.exists(scenario_path):
        raise FileNotFoundError(
            f"Scenario file not found: {scenario_path}\n"
            f'Hint: use --list to see files under "{scenario_dir}"'
        )

    base = import_module_from_path("HeuristicBaseModel", base_path)
    plots = import_module_from_path("HeuristicPlots", plots_path)
    scenario = import_module_from_path("ScenarioModule", scenario_path)

    if not hasattr(scenario, "get_data"):
        raise AttributeError(f"Scenario file must define get_data(): {scenario_path}")

    data = scenario.get_data()

    # --- Pre-run verification + plots (same as old script behavior)
    if not args.no_plots:
        plots.verify_data(data)
        plots.visualize_jobs_and_ops(data)
        plots.visualize_job_operation_membership(data)
        plots.visualize_precedence_matrix(data)

    # --- Run heuristic
    res = base.run_heuristic(data, k1=args.k1)

    # --- Feasibility check
    if hasattr(base, "check_heuristic_solution"):
        base.check_heuristic_solution(data, res)

    # --- SAME FORMAT OUTPUT
    print("\n===== Objective (Heuristic) =====")
    print(f"T_max = {res.T_max:.2f}")
    print(f"C_max = {res.C_max:.2f}")

    print("\n===== Jobs =====")
    for j in data["J"]:
        print(
            f"Job {j}: C_weld={res.C_weld[j]:.2f}, C_final={res.C_final[j]:.2f}, "
            f"T_j={res.T[j]:.2f}, d_j={data['d_j'][j]:.2f}"
        )

    print("\n===== Operations =====")
    for i in data["I"]:
        print(
            f"Op {i}: S={res.S[i]:.2f}, C={res.C[i]:.2f}, "
            f"machine={res.assign_machine[i]}, station={res.assign_station[i]}"
        )

    # --- Post-run Gantt plots
    if not args.no_plots:
        plots.plot_gantt_by_machine_colored(data, res)
        plots.plot_gantt_by_station_colored(data, res)

        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
