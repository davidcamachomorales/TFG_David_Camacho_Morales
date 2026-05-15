"""
Script auxiliar para limpiar restos de la refactorización.

Qué hace:
1. Corrige referencias a escenarios *_v4.csv que no existen.
2. Elimina definiciones duplicadas de funciones que ya se importan desde src.experiment_utils.
3. No toca el entorno de trading ni cambia lógica matemática.

Uso:
    python clean_dupes.py

Recomendación:
    Ejecutar después de revisar que DQN.py, PPO.py y A2C.py importan estas funciones:
    - aggregate_seed_results
    - get_transaction_cost
    desde src.experiment_utils.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

FILES_TO_EDIT = [
    PROJECT_ROOT / "experiments" / "DQN.py",
    PROJECT_ROOT / "experiments" / "PPO.py",
    PROJECT_ROOT / "experiments" / "A2C.py",
]

OLD_SCENARIO_FILE = "scenarios_config_1d_ablation_v4.csv"
NEW_SCENARIO_FILE = "scenarios_config_1d_ablation.csv"

FUNCTIONS_TO_REMOVE = {
    "aggregate_seed_results",
    "get_transaction_cost",
}


def remove_top_level_function(source: str, function_name: str) -> str:
    """Remove a top-level function definition by indentation, preserving the rest."""
    lines = source.splitlines(keepends=True)
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.startswith(f"def {function_name}("):
            i += 1
            while i < len(lines):
                current = lines[i]
                if current.startswith("def ") or current.startswith("class "):
                    break
                if current and not current.startswith((" ", "\t", "\n", "\r")) and current.strip():
                    break
                i += 1
            continue

        result.append(line)
        i += 1

    return "".join(result)


def ensure_no_v4_scenario_reference(source: str) -> str:
    return source.replace(OLD_SCENARIO_FILE, NEW_SCENARIO_FILE)


def main() -> None:
    for path in FILES_TO_EDIT:
        if not path.exists():
            print(f"[SKIP] No existe: {path}")
            continue

        original = path.read_text(encoding="utf-8")
        updated = ensure_no_v4_scenario_reference(original)

        for function_name in FUNCTIONS_TO_REMOVE:
            updated = remove_top_level_function(updated, function_name)

        if updated != original:
            path.write_text(updated, encoding="utf-8")
            print(f"[OK] Actualizado: {path.relative_to(PROJECT_ROOT)}")
        else:
            print(f"[OK] Sin cambios necesarios: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
