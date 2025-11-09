import argparse
import importlib.util
import inspect
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DATASET_ROOT = Path("arc-agi-2-public-dataset").resolve()


def _guard_out(path_like):
    rp = Path(path_like).resolve()
    try:
        rp.relative_to(DATASET_ROOT)
        raise SystemExit(f"[GUARD] refusing to write inside dataset: {rp}")
    except ValueError:
        try:
            rp.relative_to(SPEC_ROOT)
            raise SystemExit(f"[GUARD] refusing to write inside spec root: {rp}")
        except ValueError:
            return rp


def _ensure_grid(grid):
    if not isinstance(grid, list) or not grid:
        return [[0]]
    if isinstance(grid[0], int):
        grid = [grid]
    width = len(grid[0]) if grid and isinstance(grid[0], list) else 0
    normalized = []
    for row in grid:
        if not isinstance(row, list) or len(row) != width:
            return [[0]]
        normalized.append([int(v) if 0 <= int(v) <= 9 else 0 for v in row])
    return normalized


def load_module_from_path(py_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def to_task_map(table):
    if isinstance(table, dict):
        return table
    if isinstance(table, list):
        mapped = {}
        for item in table:
            task_id = item.get("task_id") or item.get("id")
            if task_id:
                mapped[task_id] = item.get("task") or item
        return mapped
    return {}


def sanity_first_k(module, challenges_path, k=15):
    with open(challenges_path, "r") as handle:
        blob = json.load(handle)
    tasks = list(to_task_map(blob).items())[:k]
    results = []
    for task_id, task in tasks:
        train = task["train"]
        best_score = -1
        best_label = None
        make_candidates = getattr(module, "make_candidates", None)
        if make_candidates:
            candidates = make_candidates(train)

            def score_candidate(fn):
                score = 0
                for example in train:
                    prediction = _ensure_grid(fn(_ensure_grid(example["input"])))
                    if prediction == _ensure_grid(example["output"]):
                        score += 1
                return score

            for label, fn in candidates:
                try:
                    score = score_candidate(fn)
                except Exception:
                    score = -1
                if score > best_score:
                    best_score, best_label = score, label
        else:
            best_score, best_label = -1, "NO_MAKE_CANDIDATES"

        results.append((task_id, f"{best_score}/{len(train)}", best_label))
    return results


def quick_eval(module, eval_ch_path, eval_sol_path, limit=None):
    with open(eval_ch_path, "r") as handle:
        challenges = json.load(handle)
    with open(eval_sol_path, "r") as handle:
        solutions = json.load(handle)

    tasks = list(to_task_map(challenges).items())
    if limit:
        tasks = tasks[:limit]
    hits = 0
    total = 0
    tasks_with_hit = 0
    for task_id, task in tasks:
        train = task["train"]
        task_hit = False
        for index, pair in enumerate(task["test"]):
            predictions = module.solve_arc_task(train, pair["input"], topk=2)
            ground_truth = solutions[task_id][index]
            total += 1
            if isinstance(ground_truth, list) and ground_truth and isinstance(ground_truth[0], list):
                normalized_gt = [_ensure_grid(grid) for grid in ground_truth]
            else:
                normalized_gt = [_ensure_grid(ground_truth)]
            if any(_ensure_grid(prediction) in normalized_gt for prediction in predictions):
                hits += 1
                task_hit = True
        if task_hit:
            tasks_with_hit += 1
    return hits, total, tasks_with_hit, len(tasks)


@dataclass
class HitMissRow:
    task_id: str
    ham: float
    exact: bool


def parse_miss_hit_lines(log_text: str) -> list[HitMissRow]:
    rows: list[HitMissRow] = []
    pattern = re.compile(r"\[(EM|MISS|HIT)\]\s+(\S+)\s+hamming=([0-9.]+)")
    for line in log_text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        tag, task_id, ham = match.groups()
        try:
            ham_val = float(ham)
            rows.append(HitMissRow(task_id=task_id, ham=ham_val, exact=(tag.upper() in {"EM", "HIT"} and ham_val == 0.0)))
        except ValueError:
            continue
    return rows


def write_hamming_csv(log_text: str) -> None:
    rows = parse_miss_hit_lines(log_text)
    if not rows:
        print("[HAMMING] no rows — skipping write.")
        return
    path = _guard_out("/tmp/hamming_eval.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("task_id,ham,exact\n")
        for row in rows:
            handle.write(f"{row.task_id},{row.ham:.6f},{int(row.exact)}\n")
    print(f"[HAMMING] wrote {path} rows={len(rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", required=True, help="Path to ril_solver_patch.py or ril/solver.py")
    parser.add_argument("--eval_ch", default="data/arc-agi_evaluation_challenges.json")
    parser.add_argument("--eval_sol", default="data/arc-agi_evaluation_solutions.json")
    parser.add_argument("--test_ch", default="data/arc-agi_test_challenges.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log", default=None, help="Path to log file for HIT/MISS parsing")
    arguments = parser.parse_args()

    module = load_module_from_path(arguments.solver, "solver_mod")

    print("[Where] solve_arc_task from:", inspect.getsourcefile(module.solve_arc_task))

    rows = sanity_first_k(module, arguments.eval_ch, k=15)
    print("\nSanity — first 15 eval tasks (best train-fit / #train, best label):")
    for task_id, score, label in rows:
        print(f"  {task_id}: {score}  [{label}]")

    hits, total, tasks_with_hit, num_tasks = quick_eval(
        module,
        arguments.eval_ch,
        arguments.eval_sol,
        limit=arguments.limit,
    )
    accuracy = hits / total if total else 0.0
    print(
        f"\n[EVAL] total={total} hits={hits} acc={accuracy:.4f}, "
        f"tasks_with>=1_hit={tasks_with_hit}/{num_tasks}"
    )
    if arguments.log:
        try:
            with open(arguments.log, "r") as log_handle:
                write_hamming_csv(log_handle.read())
        except Exception as exc:
            print(f"[WARN] Failed to write hamming.csv: {type(exc).__name__}:{exc}")
