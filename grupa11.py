"""Grupa 11 — udaljenosti / sličnost nizova (Loto 7/39)."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4648_k55.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def presence_vec(draw) -> np.ndarray:
    v = np.zeros(FRONT_N, dtype=float)
    for n in draw:
        v[int(n) - 1] = 1.0
    return v


def hamming(a, b) -> float:
    return float(np.sum(presence_vec(a) != presence_vec(b)))


def jaccard(a, b) -> float:
    sa, sb = set(map(int, a)), set(map(int, b))
    u = len(sa | sb)
    return len(sa & sb) / u if u else 0.0


def dice(a, b) -> float:
    sa, sb = set(map(int, a)), set(map(int, b))
    return 2 * len(sa & sb) / (len(sa) + len(sb)) if (sa or sb) else 0.0


def cosine(a, b) -> float:
    va, vb = presence_vec(a), presence_vec(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))


def edit_distance(a, b) -> int:
    """Levenshtein na sortiranim 7-torkama (kao stringovi brojeva)."""
    x = [int(v) for v in a]
    y = [int(v) for v in b]
    n, m = len(x), len(y)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if x[i - 1] == y[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[n, m])


def dtw(a, b) -> float:
    """DTW na sortiranim pozicijama (7 brojeva kao niz)."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    n, m = len(x), len(y)
    inf = 1e18
    dp = np.full((n + 1, m + 1), inf)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def soft_dtw(a, b, gamma: float = 1.0) -> float:
    """Soft-DTW (Cuturi) na 7-torkama."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    n, m = len(x), len(y)
    r = np.full((n + 2, m + 2), np.inf)
    r[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            # softmin
            vals = np.array([r[i - 1, j], r[i, j - 1], r[i - 1, j - 1]])
            soft = -gamma * np.log(np.sum(np.exp(-vals / gamma)) + 1e-300)
            r[i, j] = cost + soft
    return float(r[n, m])


def paa(series: np.ndarray, n_segments: int = 20) -> np.ndarray:
    """Piecewise Aggregate Approximation."""
    t = len(series)
    edges = np.linspace(0, t, n_segments + 1).astype(int)
    return np.array([series[edges[i] : edges[i + 1]].mean() for i in range(n_segments)])


def sax(series: np.ndarray, n_segments: int = 20, alphabet: int = 5) -> str:
    """SAX: PAA + z-score + simboli."""
    p = paa(series, n_segments)
    p = (p - p.mean()) / (p.std() + 1e-12)
    # breakpoints for N(0,1) approx equal bins
    # alphabet=5 → quantiles
    qs = np.linspace(0, 1, alphabet + 1)[1:-1]
    # inverse erf approx via np — use empirical on standard normal samples
    rng = np.random.default_rng(SEED)
    ref = np.sort(rng.normal(size=50_000))
    br = [ref[int(q * (len(ref) - 1))] for q in qs]
    letters = []
    for v in p:
        idx = 0
        while idx < len(br) and v > br[idx]:
            idx += 1
        letters.append(chr(ord("a") + idx))
    return "".join(letters)


def matrix_profile_sum(draws: np.ndarray, window: int = 32) -> dict:
    """
    Matrix profile na sumi kola (ceo CSV), Euclid na z-norm prozorima.
    Vektorisano po chunk-ovima.
    """
    s = draws.sum(axis=1).astype(float)
    t = len(s)
    if t < window * 2:
        return {"mp_min": None, "mp_argmin": None, "motif_pair": None}
    n_win = t - window + 1
    # shape (n_win, window)
    idx = np.arange(window)[None, :] + np.arange(n_win)[:, None]
    wins = s[idx]
    mu = wins.mean(axis=1, keepdims=True)
    sd = wins.std(axis=1, keepdims=True) + 1e-12
    wins = (wins - mu) / sd

    mp = np.full(n_win, np.inf)
    mpi = -np.ones(n_win, dtype=int)
    excl = window // 2
    chunk = 64
    for i0 in range(0, n_win, chunk):
        i1 = min(n_win, i0 + chunk)
        block = wins[i0:i1]  # c × w
        # distances to all windows: (c, n_win)
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b ; z-norm → ||a||^2 = window
        dots = block @ wins.T
        d = np.sqrt(np.maximum(0.0, 2 * window - 2 * dots))
        for bi, i in enumerate(range(i0, i1)):
            lo = max(0, i - excl)
            hi = min(n_win, i + excl + 1)
            d[bi, lo:hi] = np.inf
            j = int(np.argmin(d[bi]))
            mp[i] = float(d[bi, j])
            mpi[i] = j
    i0 = int(np.argmin(mp))
    return {
        "mp_min": float(mp[i0]),
        "mp_argmin": i0,
        "motif_pair": (i0, int(mpi[i0])),
        "mp_mean": float(mp.mean()),
        "mp_max": float(mp.max()),
    }


def neighbor_stats(draws: np.ndarray) -> dict:
    """Sličnost uzastopnih kola (ceo CSV) — Hamming/Jaccard/Dice/cosine/edit/DTW."""
    ham, jac, dic, cos, ed, dt = [], [], [], [], [], []
    for i in range(len(draws) - 1):
        a, b = draws[i], draws[i + 1]
        ham.append(hamming(a, b))
        jac.append(jaccard(a, b))
        dic.append(dice(a, b))
        cos.append(cosine(a, b))
        ed.append(edit_distance(a, b))
        dt.append(dtw(a, b))
    return {
        "hamming_mean": float(np.mean(ham)),
        "jaccard_mean": float(np.mean(jac)),
        "dice_mean": float(np.mean(dic)),
        "cosine_mean": float(np.mean(cos)),
        "edit_mean": float(np.mean(ed)),
        "dtw_mean": float(np.mean(dt)),
    }


def most_similar_to_last(draws: np.ndarray, top_k: int = 10) -> dict:
    """Najsličnija istorijska kola poslednjem (Jaccard + DTW)."""
    last = draws[-1]
    rows = []
    for i in range(len(draws) - 1):
        a = draws[i]
        rows.append(
            (
                i,
                jaccard(a, last),
                dice(a, last),
                cosine(a, last),
                -dtw(a, last),
                -edit_distance(a, last),
                -hamming(a, last),
            )
        )
    # rang po jaccard pa dtw
    rows.sort(key=lambda t: (-t[1], -t[4], t[0]))
    top = []
    for i, jac, dic, cos, neg_dtw, neg_ed, neg_ham in rows[:top_k]:
        top.append(
            {
                "idx": int(i),
                "draw": [int(x) for x in draws[i].tolist()],
                "jaccard": float(jac),
                "dice": float(dic),
                "cosine": float(cos),
                "dtw": float(-neg_dtw),
                "edit": int(-neg_ed),
                "hamming": int(-neg_ham),
            }
        )
    return {"last": [int(x) for x in last.tolist()], "top": top}


def learn_next_rule(draws: np.ndarray) -> dict:
    """
    Pravilo next iz grupe 11:
    nađi istorijska kola najsličnija last (Jaccard/Dice/cosine),
    uzmi njihova SLEDEĆA kola → frekvencija brojeva = skor.
    """
    last = draws[-1]
    sims = []
    for i in range(len(draws) - 1):
        a = draws[i]
        sim = 0.5 * jaccard(a, last) + 0.3 * dice(a, last) + 0.2 * cosine(a, last)
        # manji DTW / edit = bolje
        sim -= 0.01 * dtw(a, last)
        sim -= 0.05 * edit_distance(a, last)
        sims.append((i, sim))
    sims.sort(key=lambda t: (-t[1], t[0]))
    top_idx = [i for i, _ in sims[:80]]

    # next draws after similar
    nxt_counts = Counter()
    for i in top_idx:
        for n in draws[i + 1].tolist():
            nxt_counts[int(n)] += 1
    max_c = max(nxt_counts.values()) if nxt_counts else 1
    global_f = Counter(draws.reshape(-1).tolist())
    max_g = max(global_f.values()) if global_f else 1

    number_score = {}
    for y in range(1, FRONT_N + 1):
        number_score[y] = 1.5 * (nxt_counts.get(y, 0) / max_c) + 0.2 * (global_f.get(y, 0) / max_g)

    # target sum from those next draws
    next_sums = [float(draws[i + 1].sum()) for i in top_idx]
    return {
        "number_score": number_score,
        "last_draw": [int(x) for x in last.tolist()],
        "target_sum": float(np.mean(next_sums)),
        "n_neighbors": len(top_idx),
        "top_sim": float(sims[0][1]) if sims else 0.0,
    }


def _combo_fit(combo: list[int], rule: dict) -> float:
    score = sum(rule["number_score"][x] for x in combo)
    score -= 0.02 * abs(sum(combo) - rule["target_sum"])
    return score


def predict_next_from_rule(draws: np.ndarray, rule: dict | None = None) -> list[int]:
    if rule is None:
        rule = learn_next_rule(draws)
    ranked = sorted(rule["number_score"], key=lambda n: (-rule["number_score"][n], n))
    best = None
    best_fit = -1e18
    for start in range(0, min(20, FRONT_N - FRONT_SELECT + 1)):
        base = sorted(ranked[start : start + FRONT_SELECT])
        for repl in ranked[:28]:
            cand = sorted(set(base[1:] + [repl]))
            if len(cand) != FRONT_SELECT:
                continue
            fit = _combo_fit(cand, rule)
            if fit > best_fit:
                best_fit = fit
                best = cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_grupa11(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    print(f"CSV: {csv_path.name}")
    print(f"Kola: {len(draws)} | seed={SEED} | 7/39 | grupa11")
    print()

    print("=== sličnost uzastopnih kola (mean) ===")
    print(neighbor_stats(draws))
    print()

    print("=== najsličnija last (top5) ===")
    ms = most_similar_to_last(draws, top_k=5)
    print("last:", ms["last"])
    for row in ms["top"]:
        print(row)
    print()

    sums = draws.sum(axis=1).astype(float)
    print("=== PAA/SAX sume ===")
    print({"paa20": [round(float(v), 2) for v in paa(sums, 20)], "sax20": sax(sums, 20, 5)})
    print()

    print("=== matrix profile (suma, window=32) ===")
    # subsample stride inside for speed: use every point but window 32 on full — can be slow
    # optimize: compute on full with step in inner loop already O(n^2); n~4600 → heavy
    # use window on downsampled? User wants full CSV — series is full; MP brute is slow.
    # Use smaller effective compare: every 2nd window start still covers full series values.
    mp = matrix_profile_sum(draws, window=32)
    print(mp)
    print()

    print("=== soft-DTW last vs mean-prototype ===")
    proto = np.round(draws.mean(axis=0)).astype(int)
    # unique sorted-ish
    proto = sorted(set(max(1, min(FRONT_N, int(x))) for x in proto))
    while len(proto) < FRONT_SELECT:
        for n in range(1, FRONT_N + 1):
            if n not in proto:
                proto.append(n)
                break
    proto = sorted(proto)[:FRONT_SELECT]
    print({"proto": proto, "soft_dtw": soft_dtw(draws[-1], proto), "dtw": dtw(draws[-1], proto)})
    print()

    print("=== pravilo → next (grupa 11) ===")
    rule = learn_next_rule(draws)
    combo = predict_next_from_rule(draws, rule)
    print(
        "rule:",
        {
            "last_draw": rule["last_draw"],
            "target_sum": round(rule["target_sum"], 2),
            "n_neighbors": rule["n_neighbors"],
            "top_sim": round(rule["top_sim"], 4),
        },
    )
    print("next:", combo)


if __name__ == "__main__":
    run_grupa11()


"""
11. Udaljenosti / sličnost nizova
Hamming, Jaccard, Dice, cosine, edit distance, DTW, soft-DTW, SAX, PAA, matrix profile
"""



"""
CSV: loto7_4648_k55.csv
Kola: 4648 | seed=39 | 7/39 | grupa11

=== sličnost uzastopnih kola (mean) ===
{'hamming_mean': 11.53260167850226, 'jaccard_mean': 0.10266088071123577, 'dice_mean': 0.1762427372498386, 'cosine_mean': 0.17624273724981337, 'edit_mean': 6.463955239939746, 'dtw_mean': 26.708844415752097}

=== najsličnija last (top5) ===
last: [3, 7, 12, 13, 18, 24, 29]
{'idx': 2995, 'draw': [2, 3, 13, 18, 24, 28, 29], 'jaccard': 0.5555555555555556, 'dice': 0.7142857142857143, 'cosine': 0.7142857142856122, 'dtw': 7.0, 'edit': 4, 'hamming': 4}
{'idx': 4155, 'draw': [1, 3, 12, 13, 20, 24, 29], 'jaccard': 0.5555555555555556, 'dice': 0.7142857142857143, 'cosine': 0.7142857142856122, 'dtw': 8.0, 'edit': 3, 'hamming': 4}
{'idx': 1373, 'draw': [2, 3, 7, 13, 18, 29, 39], 'jaccard': 0.5555555555555556, 'dice': 0.7142857142857143, 'cosine': 0.7142857142856122, 'dtw': 17.0, 'edit': 4, 'hamming': 4}
{'idx': 1021, 'draw': [4, 8, 12, 13, 18, 26, 29], 'jaccard': 0.4, 'dice': 0.5714285714285714, 'cosine': 0.5714285714284897, 'dtw': 4.0, 'edit': 3, 'hamming': 6}
{'idx': 3605, 'draw': [3, 6, 12, 18, 23, 29, 31], 'jaccard': 0.4, 'dice': 0.5714285714285714, 'cosine': 0.5714285714284897, 'dtw': 5.0, 'edit': 4, 'hamming': 6}

=== PAA/SAX sume ===
{'paa20': [140.77, 138.86, 141.05, 143.14, 141.98, 141.27, 137.7, 139.41, 140.37, 141.03, 140.51, 143.68, 136.83, 138.06, 139.31,140.74, 141.57, 140.87, 143.81, 137.67], 'sax20': 'cbdeddabcdceaabcdcea'}

=== matrix profile (suma, window=32) ===
{'mp_min': 3.990164717138369, 'mp_argmin': 1060, 'motif_pair': (1060, 3673), 'mp_mean': 5.055730216724191, 'mp_max': 5.664741548223058}

=== soft-DTW last vs mean-prototype ===
{'proto': [5, 10, 15, 20, 25, 30, 35], 'soft_dtw': 57.992744922798224, 'dtw': 18.0}

=== pravilo → next (grupa 11) ===
rule: {'last_draw': [3, 7, 12, 13, 18, 24, 29], 'target_sum': 136.22, 'n_neighbors': 80, 'top_sim': 0.4049}
next: [5, x, 20, y, 23, z, 32]
"""
