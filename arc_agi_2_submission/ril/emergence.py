# ril/emergence.py
# Lightweight event/hypothesis layer for ARC. Safe, gated, and fallback-friendly.

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .np_compat import NP, USE_NUMPY

if USE_NUMPY:  # optional dependency
    np = NP  # type: ignore
else:  # pragma: no cover - exercised when NumPy disabled/missing
    np = None  # type: ignore
import csv, pathlib, time

# -----------------------
# Feature flags / knobs
# -----------------------
ENABLE_EVENTS = os.getenv("ENABLE_EVENTS", "0") == "1"
NUMPY_AVAILABLE = USE_NUMPY
EVENT_MIN_COVERAGE = float(os.getenv("EVENT_MIN_COVERAGE", "0.80"))  # conservative but not too strict
EVENT_MAX_OPS = int(os.getenv("EVENT_MAX_OPS", "3"))
EVENT_LOG_ENABLE = os.getenv("EVENT_LOG_ENABLE", "0") == "1"
EVENT_LOG_PATH = os.getenv("EVENT_LOG_PATH", "/kaggle/working/event_trace.csv")
MDL_LAMBDA = float(os.getenv("EVENT_MDL_LAMBDA", "0.5"))
MDL_MU = float(os.getenv("EVENT_MDL_MU", "0.1"))
BEAM_K = int(os.getenv("EVENT_BEAM_K", "8"))

# -----------------------
# Utilities
# -----------------------

def detect_bg_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)]) if len(vals) else 0

def connected_components(grid: np.ndarray) -> List[np.ndarray]:
    """
    Return boolean masks of components. Tries scipy.ndimage.label; falls back to one mask per color.
    """
    try:
        from scipy.ndimage import label  # optional
        masks: List[np.ndarray] = []
        for c in np.unique(grid):
            color_mask = (grid == c)
            if not color_mask.any():
                continue
            lab, n = label(color_mask)  # 4-neighb default
            for k in range(1, n + 1):
                comp = (lab == k)
                if comp.any():
                    masks.append(comp.astype(bool))
        return masks
    except Exception:
        return [(grid == c) for c in np.unique(grid) if (grid == c).any()]

def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    if inter == 0:
        return 0.0
    union = (a | b).sum()
    return float(inter) / float(union + 1e-9)

def centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return (0.0, 0.0)
    return (float(np.mean(ys)), float(np.mean(xs)))

# -----------------------
# Event extraction
# -----------------------

@dataclass
class Cards:
    new_colors: List[int]
    dead_colors: List[int]
    matches: List[Tuple[np.ndarray, np.ndarray, float, int, int]]  # (min, mout, iou, color_in, color_out)
    global_dxdy: Optional[Tuple[int, int]]

def extract_events(inp: np.ndarray, out: np.ndarray) -> Cards:
    pins = set(np.unique(inp))
    pouts = set(np.unique(out))
    new_colors = list(pouts - pins)
    dead_colors = list(pins - pouts)

    comps_in = connected_components(inp)
    comps_out = connected_components(out)

    matches: List[Tuple[np.ndarray, np.ndarray, float, int, int]] = []
    for mi in comps_in:
        ci = int(np.bincount(inp[mi].ravel()).argmax()) if mi.any() else 0
        best = None; best_iou = 0.0; best_co = None
        for mo in comps_out:
            i = iou(mi, mo)
            if i > best_iou:
                best_iou = i
                if mo.any():
                    best_co = int(np.bincount(out[mo].ravel()).argmax())
                    best = mo
        if best is not None:
            matches.append((mi, best, best_iou, ci, best_co if best_co is not None else ci))

    # try to infer a common translation from matched comps
    dxy_votes: Dict[Tuple[int, int], int] = {}
    for mi, mo, iouv, ci, co in matches:
        if iouv < 0.4:  # not strong enough to trust
            continue
        (y1, x1) = centroid(mi)
        (y2, x2) = centroid(mo)
        dx = int(round(x2 - x1))
        dy = int(round(y2 - y1))
        dxy_votes[(dx, dy)] = dxy_votes.get((dx, dy), 0) + 1
    global_dxdy = None
    if dxy_votes:
        (dx, dy), cnt = max(dxy_votes.items(), key=lambda kv: kv[1])
        if cnt >= 2:  # seen at least twice
            global_dxdy = (dx, dy)

    return Cards(
        new_colors=new_colors,
        dead_colors=dead_colors,
        matches=matches,
        global_dxdy=global_dxdy,
    )

# -----------------------
# Hypotheses / operators
# -----------------------

@dataclass
class Hypothesis:
    ops: List[Tuple[str, Dict]]  # [("OP", {params})]

def looks_rot90(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.shape[::-1] != out.shape:
        return False
    return (np.rot90(inp) == out).mean() > 0.9

def infer_recolor_map(train_pairs: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Dict[int, int]:
    votes: Dict[int, Dict[int, int]] = {}
    for inp, out in train_pairs:
        comps_in = connected_components(inp)
        comps_out = connected_components(out)
        for mi in comps_in:
            if not mi.any():
                continue
            ci = int(np.bincount(inp[mi].ravel()).argmax())
            best = None; best_iou = 0.0; best_co = None
            for mo in comps_out:
                i = iou(mi, mo)
                if i > best_iou:
                    best_iou = i
                    if mo.any():
                        best_co = int(np.bincount(out[mo].ravel()).argmax())
                        best = mo
            if best is not None and best_iou > 0.5 and best_co is not None:
                votes.setdefault(ci, {}).setdefault(best_co, 0)
                votes[ci][best_co] += 1

    cmap = {}
    for src, dstcounts in votes.items():
        dst = max(dstcounts.items(), key=lambda kv: kv[1])[0]
        if src != dst:
            cmap[src] = dst
    return cmap

def infer_emerge_rect_and_color(train_pairs: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[Optional[Tuple[int,int,int,int]], Optional[int]]:
    rects = []; colors = []
    for inp, out in train_pairs:
        pins = set(np.unique(inp)); pouts = set(np.unique(out))
        news = list(pouts - pins)
        if not news:
            return None, None
        best_mask = None; best_area = 0; best_c = None
        for c in news:
            mask = (out == c)
            area = int(mask.sum())
            if area > best_area:
                best_area = area; best_mask = mask; best_c = c
        ys, xs = np.where(best_mask)
        y0, x0, y1, x1 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
        rects.append((y0, x0, y1 - y0, x1 - x0))
        colors.append(best_c)
    rects = np.array(rects)
    med = np.median(rects, axis=0).astype(int).tolist()
    vals, cnts = np.unique(np.array(colors), return_counts=True)
    col = int(vals[np.argmax(cnts)])
    return tuple(med), col

def seed_hypotheses(signals: Dict, cards: Cards) -> List[Hypothesis]:
    hyps: List[Hypothesis] = []

    # rotation (cheap, decisive)
    if signals.get("looks_rot90_all", False):
        hyps.append(Hypothesis([("ROT90", {})]))

    # global translation
    if cards.global_dxdy is not None:
        dx, dy = cards.global_dxdy
        hyps.append(Hypothesis([("RELOCATE", {"dx": dx, "dy": dy})]))

    # recolor (learned)
    hyps.append(Hypothesis([("RECOLOR_BY_MAP", {})]))

    # border fill (often appears)
    hyps.append(Hypothesis([("FILL_BORDER", {})]))

    # emergence + relocate + recolor (small chain)
    hyps.append(Hypothesis([("EMERGE_DOMINANT_RECT", {}), ("RELOCATE", {}), ("RECOLOR_BY_MAP", {})]))

    # cap beam
    return hyps[:BEAM_K]

def unify(h: Hypothesis, train_pairs: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[bool, Dict]:
    params: Dict = {}

    # rotation viability
    if any(op[0] == "ROT90" for op in h.ops):
        if not all(looks_rot90(i, o) for (i, o) in train_pairs):
            return False, {}

    # recolor map
    if any(op[0] == "RECOLOR_BY_MAP" for op in h.ops):
        cmap = infer_recolor_map(train_pairs)
        if not cmap:
            return False, {}
        params["cmap"] = cmap

    # relocate: infer dx,dy if missing
    for i, (name, kw) in enumerate(h.ops):
        if name == "RELOCATE" and ("dx" not in kw or "dy" not in kw):
            dxy = []
            for inp, out in train_pairs:
                c = extract_events(inp, out)
                if c.global_dxdy is not None:
                    dxy.append(c.global_dxdy)
            if not dxy:
                return False, {}
            # majority vote
            hist: Dict[Tuple[int,int], int] = {}
            for p in dxy:
                hist[p] = hist.get(p, 0) + 1
            dx, dy = max(hist.items(), key=lambda kv: kv[1])[0]
            h.ops[i] = ("RELOCATE", {"dx": int(dx), "dy": int(dy)})

    # emergence params
    if any(op[0] == "EMERGE_DOMINANT_RECT" for op in h.ops):
        rect, color = infer_emerge_rect_and_color(train_pairs)
        if rect is None:
            return False, {}
        params["emerge_rect"] = rect
        params["emerge_color"] = color

    # refuse overly long chains
    if len(h.ops) > EVENT_MAX_OPS:
        return False, {}

    return True, params

def apply_ops(h: Hypothesis, grid: np.ndarray, aux: Optional[Dict] = None) -> np.ndarray:
    out = grid.copy()
    H, W = out.shape
    bg = detect_bg_color(out)
    params = aux or {}

    for name, kw in h.ops:
        if name == "ROT90":
            out = np.rot90(out)

        elif name == "RELOCATE":
            dx, dy = int(kw["dx"]), int(kw["dy"])
            pad_top = max(0, -dy); pad_left = max(0, -dx)
            pad_bot = max(0, dy);  pad_right = max(0, dx)
            padded = np.pad(out, ((pad_top, pad_bot), (pad_left, pad_right)), constant_values=bg)
            shifted = np.full_like(padded, fill_value=bg)
            ys, xs = np.where(padded != bg)
            ys2 = ys + dy; xs2 = xs + dx
            ok = (xs2 >= 0) & (ys2 >= 0) & (ys2 < padded.shape[0]) & (xs2 < padded.shape[1])
            shifted[ys2[ok], xs2[ok]] = padded[ys[ok], xs[ok]]
            out = shifted

        elif name == "RECOLOR_BY_MAP":
            cmap: Dict[int, int] = params.get("cmap", {})
            if cmap:
                # don't recolor background
                mask = out != bg
                unique_vals = np.unique(out[mask])
                for v in unique_vals:
                    if v in cmap and cmap[v] != v:
                        out[out == v] = cmap[v]

        elif name == "FILL_BORDER":
            # majority non-bg color
            nz = out[out != bg]
            if nz.size > 0:
                vals, cnts = np.unique(nz, return_counts=True)
                c = int(vals[np.argmax(cnts)])
                out[0, :] = c; out[-1, :] = c; out[:, 0] = c; out[:, -1] = c

        elif name == "EMERGE_DOMINANT_RECT":
            rect = params.get("emerge_rect")
            newc = params.get("emerge_color")
            if rect is None:
                h2, w2 = max(1, H // 3), max(1, W // 3)
                y0, x0 = (H - h2) // 2, (W - w2) // 2
                rect = (y0, x0, h2, w2)
            if newc is None:
                # choose a non-bg color
                candidate = int((bg + 1) % 10)
                if candidate == bg:
                    candidate = (candidate + 1) % 10
                newc = candidate
            y0, x0, h2, w2 = rect
            y0 = max(0, min(H - 1, y0)); x0 = max(0, min(W - 1, x0))
            y1 = max(0, min(H, y0 + h2)); x1 = max(0, min(W, x0 + w2))
            out[y0:y1, x0:x1] = newc

    return out

# -----------------------
# Scoring / selection
# -----------------------

def mdl_score(pred: np.ndarray, target: np.ndarray, h: Hypothesis, lam: float, mu: float) -> float:
    if pred.shape != target.shape:
        H = min(pred.shape[0], target.shape[0])
        W = min(pred.shape[1], target.shape[1])
        pred = pred[:H, :W]
        target = target[:H, :W]
    explained = float((pred == target).mean())
    dl = sum(1 + len(kw) for (name, kw) in h.ops)
    resid_mask = (pred != target)
    if resid_mask.any():
        vals, cnts = np.unique(target[resid_mask], return_counts=True)
        p = cnts / cnts.sum()
        Hres = float(-(p * np.log2(p + 1e-12)).sum())
    else:
        Hres = 0.0
    return explained - lam * dl - mu * Hres

def select_best_hypothesis(
    train_examples: Sequence[Dict],
    seed_hyps: List[Hypothesis],
    dbg_log: List[str],
) -> Tuple[Optional[Hypothesis], Optional[Dict]]:
    train_pairs = [(np.array(tr["input"]), np.array(tr["output"])) for tr in train_examples]
    best_h, best_params, best_score = None, None, -1e9

    for h in seed_hyps:
        ok, params = unify(h, train_pairs)
        if not ok:
            dbg_log.append(f"[EVENT] reject unify: {h.ops}")
            continue
        scores = []
        weights = []
        coverages = []
        for inp, out in train_pairs:
            pred = apply_ops(h, inp, aux=params)
            s = mdl_score(pred, out, h, MDL_LAMBDA, MDL_MU)
            scores.append(s)
            weights.append(out.size)
            # coverage as pixel agreement
            if pred.shape != out.shape:
                H = min(pred.shape[0], out.shape[0])
                W = min(pred.shape[1], out.shape[1])
                cov = (pred[:H, :W] == out[:H, :W]).mean()
            else:
                cov = (pred == out).mean()
            coverages.append(float(cov))
        avg_score = float(np.average(scores, weights=weights)) if scores else -1e9
        avg_cov = float(np.average(coverages, weights=weights)) if coverages else 0.0
        dbg_log.append(f"[EVENT] cand {h.ops} | cov={avg_cov:.3f} | mdl={avg_score:.3f}")
        if avg_cov >= EVENT_MIN_COVERAGE and avg_score > best_score:
            best_h, best_params, best_score = Hypothesis(list(h.ops)), dict(params), avg_score

    return best_h, best_params

def _score_on_trains(train_pairs, h: Hypothesis, params: Dict) -> Tuple[float, float]:
    """Return (avg_coverage, avg_mdl) over train pairs."""
    scores, covs, weights = [], [], []
    for inp, out in train_pairs:
        pred = apply_ops(h, inp, aux=params)
        # coverage
        if pred.shape != out.shape:
            H = min(pred.shape[0], out.shape[0]); W = min(pred.shape[1], out.shape[1])
            cov = (pred[:H, :W] == out[:H, :W]).mean()
        else:
            cov = (pred == out).mean()
        # mdl
        s = mdl_score(pred, out, h, MDL_LAMBDA, MDL_MU)
        scores.append(float(s)); covs.append(float(cov)); weights.append(out.size)
    avg_cov = float(np.average(covs, weights=weights)) if covs else 0.0
    avg_mdl = float(np.average(scores, weights=weights)) if scores else -1e9
    return avg_cov, avg_mdl

def _eventsafe_append_csv(row: Dict[str, object]) -> None:
    """Append one row to EVENT_LOG_PATH if logging enabled; never raise to caller."""
    if not EVENT_LOG_ENABLE:
        return
    try:
        p = pathlib.Path(EVENT_LOG_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        file_exists = p.exists()
        with p.open("a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "ts",
                    "task_id",
                    "picked_ops",
                    "params",
                    "avg_coverage",
                    "avg_mdl",
                    "test_shape",
                    "notes",
                ],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        # keep totally silent in competition run
        pass


# -----------------------
# Public entrypoint
# -----------------------

def solve_with_events(train_examples: Sequence[Dict], test_input: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Returns (prediction_or_None, debug_info).
    If feature flag disabled or no viable hypothesis, returns (None, debug).
    """
    debug: Dict = {"enabled": ENABLE_EVENTS and NUMPY_AVAILABLE, "log": []}
    if not ENABLE_EVENTS:
        debug["log"].append("[EVENT] disabled via ENABLE_EVENTS=0")
        return None, debug

    if not NUMPY_AVAILABLE:
        debug["log"].append("[EVENT] disabled: numpy unavailable")
        return None, debug

    # signals from trains
    train_pairs = [(np.array(tr["input"]), np.array(tr["output"])) for tr in train_examples]
    cards_all = [extract_events(i, o) for (i, o) in train_pairs]
    signals = {
        "looks_rot90_all": all(looks_rot90(i, o) for (i, o) in train_pairs),
    }
    debug["log"].append(f"[EVENT] signals={signals}")

    # seed hyps from aggregate cards (use the first for dx/dy etc.; seed is generic)
    seed = seed_hypotheses(signals, cards_all[0] if cards_all else Cards([], [], [], None))
    best_h, best_params = select_best_hypothesis(train_examples, seed, debug["log"])

    if best_h is None:
        debug["log"].append("[EVENT] no hypothesis met coverage/mdl threshold; fallback")
        return None, debug

    # produce final prediction
    pred = apply_ops(best_h, np.array(test_input), aux=best_params)
    debug["picked_ops"] = best_h.ops
    debug["params"] = best_params

    # compute train-set stats for the chosen hypothesis (for the CSV)
    train_pairs = [(np.array(tr["input"]), np.array(tr["output"])) for tr in train_examples]
    avg_cov, avg_mdl = _score_on_trains(train_pairs, best_h, best_params)
    debug["log"].append(f"[EVENT] picked_ops={best_h.ops} params={best_params} cov={avg_cov:.3f} mdl={avg_mdl:.3f}")

    # opportunistic CSV write (guarded by env flag)
    try:
        _eventsafe_append_csv({
            "ts": int(time.time()),
            "task_id": os.getenv("ARC_TASK_ID", ""),     # you can set this per loop; empty is fine
            "picked_ops": repr(best_h.ops),
            "params": repr(best_params),
            "avg_coverage": round(avg_cov, 6),
            "avg_mdl": round(avg_mdl, 6),
            "test_shape": f"{np.array(test_input).shape}",
            "notes": "events_v2",
        })
    except Exception:
        pass

    return pred, debug
