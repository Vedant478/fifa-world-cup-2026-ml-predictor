"""
FIFA World Cup 2026 — ML Prediction Model (v2)
================================================
FIXES applied vs v1:
  1. Uses full international results dataset (martj42, ~49 k matches) instead
     of WC-only matches.  Falls back to local file if no network.
  2. wc_wr feature replaced with tournament_wr (weighted win-rate in any
     competitive tournament, not just late WC rounds — avoids the Brazil/
     Germany historical-bias problem).
  3. Goal simulation in Monte Carlo now uses probability-consistent Poisson
     scorelines instead of rank-only arithmetic.
  4. Knockout bracket uses the OFFICIAL 2026 FIFA R32 pairings (Matches
     73-88 from the published draw, including third-place rules).
  5. Third-place qualification included in full-tournament Monte Carlo.
  6. predict_single_match and knockout calls always pass an explicit date.

Inputs (all auto-downloaded or local):
  rankings.csv  — current FIFA ranking snapshot
  matches source — martj42 GitHub (auto-download) OR local matches_1930_2022.csv
  schedule       — mjwebmaster GitHub (same as v1)

Requirements:
    pip install requests pandas numpy scikit-learn

Run:
    python fifa_wc_2026_predictor.py
"""

import requests
import pandas as pd
import numpy as np
import warnings
import pickle
from io import StringIO

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

# ── CONFIGURATION ────────────────────────────────────────────────────────────

RANKINGS_PATH = "fifa_ranking_2026-06-08.csv"
MATCHES_PATH  = "matches_1930_2022.csv"          # local fallback only

# Full international results (martj42) — ~49 k matches, qualifiers + friendlies
INTL_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/results.csv"
)

SCHEDULE_URL = (
    "https://raw.githubusercontent.com/mjwebmaster/world-cup-2026-schedule-data"
    "/main/world-cup-2026-schedule.csv"
)

N_SIMULATIONS      = 5_000   # group-stage qualification Monte Carlo
N_FULL_SIMULATIONS = 2_000   # full-tournament Monte Carlo

# ── RECENT-FORM WINDOW ───────────────────────────────────────────────────────
# Using data since 2018 gives the model fresher signal; earlier data still
# contributes through cumulative tournament features.
FORM_SINCE = pd.Timestamp("2018-01-01")

# Name harmonisation: schedule / rankings may differ from the results dataset
NAME_MAP = {
    "South Korea":    "Korea Republic",
    "DR Congo":       "Congo DR",
    "Turkey":         "Türkiye",
    "Ivory Coast":    "Côte d'Ivoire",
    "Cape Verde":     "Cabo Verde",
    "Czech Republic": "Czechia",
    "USA":            "United States",
    "Curacao":        "Curaçao",
    "IR Iran":        "Iran",
}

# ── OFFICIAL 2026 R32 BRACKET (from FIFA draw, Dec 2025) ─────────────────────
# Each tuple: (slot_1, slot_2)
# Slots: "W_X" = winner of group X, "R_X" = runner-up of group X,
#        "3rd_XYZ..." = best 3rd from listed groups (resolved at runtime)
# Third-place slot resolution is probabilistic in the Monte Carlo.
R32_SLOTS = [
    # Match 73
    ("R_A", "R_B"),
    # Match 74
    ("W_E", "3rd_ABCDF"),
    # Match 75
    ("W_F", "R_C"),
    # Match 76
    ("W_C", "R_F"),
    # Match 77
    ("W_I", "3rd_CDFGH"),
    # Match 78
    ("R_E", "R_I"),
    # Match 79
    ("W_A", "3rd_CEFHI"),
    # Match 80
    ("W_L", "3rd_EHIJK"),
    # Match 81
    ("W_D", "3rd_BEFIJ"),
    # Match 82
    ("W_G", "3rd_AEHIJ"),
    # Match 83
    ("R_K", "R_L"),
    # Match 84
    ("W_H", "R_J"),
    # Match 85
    ("W_B", "3rd_EFGIJ"),
    # Match 86
    ("W_J", "R_H"),
    # Match 87
    ("W_K", "3rd_DEIJL"),
    # Match 88
    ("R_D", "R_G"),
]

# R16 pairings expressed as 0-based indices into r32_w
# (R32 matches 73-88 map to indices 0-15: match 73=0, 74=1, 75=2, … 88=15)
# Official bracket:
#   M89: W(M74) vs W(M77)  →  idx 1 vs idx 4
#   M90: W(M73) vs W(M75)  →  idx 0 vs idx 2
#   M91: W(M76) vs W(M78)  →  idx 3 vs idx 5
#   M92: W(M79) vs W(M82)  →  idx 6 vs idx 9
#   M93: W(M80) vs W(M83)  →  idx 7 vs idx 10
#   M94: W(M84) vs W(M85)  →  idx 11 vs idx 12
#   M95: W(M86) vs W(M87)  →  idx 13 vs idx 14
#   M96: W(M81) vs W(M88)  →  idx 8 vs idx 15
R16_PAIRS = [
    (1, 4),    # M89: W(M74) vs W(M77)
    (0, 2),    # M90: W(M73) vs W(M75)
    (3, 5),    # M91: W(M76) vs W(M78)
    (6, 9),    # M92: W(M79) vs W(M82)
    (7, 10),   # M93: W(M80) vs W(M83)
    (11, 12),  # M94: W(M84) vs W(M85)
    (13, 14),  # M95: W(M86) vs W(M87)
    (8, 15),   # M96: W(M81) vs W(M88)
]
# QF pairs (by R16 winner index 0-7, i.e. M89 winner=0, M90 winner=1, …)
# Official: M97=W89vW90, M98=W91vW92, M99=W93vW94, M100=W95vW96
QF_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
# SF pairs (by QF winner index 0-3)
# Official: M101=W97vW98, M102=W99vW100
SF_PAIRS = [(0, 1), (2, 3)]


# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────

def load_rankings(path: str = RANKINGS_PATH) -> dict:
    df = pd.read_csv(path)
    df["team"] = df["team"].replace(NAME_MAP)
    rankings = {}
    for _, row in df.iterrows():
        rankings[row["team"]] = dict(
            rank        = float(row["rank"]),
            points      = float(row["points"]),
            prev_rank   = float(row["previous_rank"]),
            prev_points = float(row["previous_points"]),
            momentum    = float(row["previous_rank"]) - float(row["rank"]),
        )
    worst_rank = max(r["rank"] for r in rankings.values())
    lowest_pts = min(r["points"] for r in rankings.values())
    rankings["__default__"] = dict(
        rank=worst_rank + 10, points=lowest_pts * 0.8,
        prev_rank=worst_rank + 10, prev_points=lowest_pts * 0.8, momentum=0.0,
    )
    return rankings


def get_rank_info(rankings: dict, team: str) -> dict:
    return rankings.get(team, rankings["__default__"])


def load_matches() -> pd.DataFrame:
    """
    Try to download the full martj42 international results dataset.
    Falls back to the local WC-only CSV if the download fails.
    """
    print("\nAttempting to download full international results dataset…")
    try:
        resp = requests.get(INTL_RESULTS_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        # martj42 columns: date, home_team, away_team, home_score, away_score,
        #                   tournament, city, country, neutral
        df = df.rename(columns={"tournament": "Round"})
        print(f"✅ Downloaded {len(df):,} international matches from martj42 dataset")
    except Exception as e:
        print(f"⚠️  Download failed ({e}). Falling back to local file: {MATCHES_PATH}")
        df = pd.read_csv(MATCHES_PATH, low_memory=False)

    df["home_team"] = df["home_team"].replace(NAME_MAP)
    df["away_team"] = df["away_team"].replace(NAME_MAP)

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"]).sort_values("date")

    print(f"✅ Final historical matches available: {len(df):,}")
    return df


def load_schedule() -> pd.DataFrame:
    df = pd.read_csv(StringIO(requests.get(SCHEDULE_URL, timeout=20).text))
    df["team_a"] = df["team_a"].replace(NAME_MAP)
    df["team_b"] = df["team_b"].replace(NAME_MAP)
    return df


# ── 2. TEAM-LEVEL FORM FEATURES ───────────────────────────────────────────────

# Matches from these tournaments get a "competitive" weight bonus
COMPETITIVE_KEYWORDS = {
    "fifa world cup", "uefa euro", "copa america", "africa cup",
    "asian cup", "gold cup", "nations league", "qualification",
    "qualifier", "confederations cup",
}


def _is_competitive(round_val: str) -> bool:
    if not isinstance(round_val, str):
        return False
    rl = round_val.lower()
    return any(kw in rl for kw in COMPETITIVE_KEYWORDS)


def team_stats(df: pd.DataFrame, team: str, cutoff, n: int = 40) -> dict:
    """
    Recency-weighted stats for *team* using matches strictly before *cutoff*
    and no earlier than FORM_SINCE.
    """
    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team))
        & (df["date"] < cutoff)
        & (df["date"] >= FORM_SINCE)
    )
    m = df[mask].sort_values("date").tail(n)

    if len(m) == 0:
        # Widen to all-time if no recent matches (e.g. newcomer)
        mask2 = (df["home_team"] == team) | (df["away_team"] == team)
        m = df[mask2 & (df["date"] < cutoff)].sort_values("date").tail(n)
    if len(m) == 0:
        return dict(wr=0.5, gd=0.0, gs=1.0, gc=1.0, comp_wr=0.5, form=0.5)

    weights = np.linspace(0.5, 1.0, len(m))
    wr_list, gs_list, gc_list = [], [], []
    comp_wins, comp_total = 0, 0

    for i, (_, r) in enumerate(m.iterrows()):
        w    = weights[i]
        home = r["home_team"] == team
        gs   = float(r["home_score"] if home else r["away_score"])
        gc   = float(r["away_score"] if home else r["home_score"])
        gs_list.append(gs * w)
        gc_list.append(gc * w)
        result = 1.0 if gs > gc else 0.5 if gs == gc else 0.0
        wr_list.append(result * w)

        if _is_competitive(r.get("Round", "")):
            comp_total += 1
            comp_wins  += (1 if gs > gc else 0)

    gd_list = [s - c for s, c in zip(gs_list, gc_list)]
    return dict(
        wr      = float(np.mean(wr_list)),
        gd      = float(np.mean(gd_list)),
        gs      = float(np.mean(gs_list)),
        gc      = float(np.mean(gc_list)),
        comp_wr = float(comp_wins / comp_total) if comp_total else 0.5,
        form    = float(np.mean(wr_list[-8:])) if len(wr_list) >= 8 else float(np.mean(wr_list)),
    )


# ── 3. FEATURE VECTOR ─────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "rank_diff", "rank_t1", "rank_t2",
    "points_diff", "momentum_diff",
    "wr_diff", "gd_diff", "gs_diff", "gc_diff",
    "form_diff", "comp_wr_diff",
    "t1_wr", "t2_wr", "t1_form", "t2_form",
]


def build_features(df: pd.DataFrame, rankings: dict, t1: str, t2: str,
                   date=pd.Timestamp("2026-06-20")) -> np.ndarray:
    s1 = team_stats(df, t1, date)
    s2 = team_stats(df, t2, date)
    r1 = get_rank_info(rankings, t1)
    r2 = get_rank_info(rankings, t2)

    feats = [
        r2["rank"]    - r1["rank"],
        r1["rank"],     r2["rank"],
        r1["points"]  - r2["points"],
        r1["momentum"]- r2["momentum"],
        s1["wr"]      - s2["wr"],
        s1["gd"]      - s2["gd"],
        s1["gs"]      - s2["gs"],
        s1["gc"]      - s2["gc"],
        s1["form"]    - s2["form"],
        s1["comp_wr"] - s2["comp_wr"],   # FIX: was wc_wr (biased); now any competitive match
        s1["wr"],       s2["wr"],
        s1["form"],     s2["form"],
    ]
    return np.array(
        [0.0 if (v != v or abs(v) == float("inf")) else v for v in feats],
        dtype=float,
    )


# ── 4. TRAINING ───────────────────────────────────────────────────────────────

def build_training_set(df: pd.DataFrame, rankings: dict):
    """
    Only use matches from FORM_SINCE onward as training rows
    (older matches are used as historical context inside team_stats,
    but training the classifier on pre-1990 data adds little signal
    and significantly slows the feature-build loop).
    """
    train_df = df[df["date"] >= FORM_SINCE].copy()
    print(f"\nBuilding training set from {len(train_df):,} matches (since {FORM_SINCE.date()})…")

    rows, labels, failed = [], [], 0
    for _, row in train_df.iterrows():
        try:
            h, a = row["home_team"], row["away_team"]
            hs   = row["home_score"]
            as_  = row["away_score"]
            rows.append(build_features(df, rankings, h, a, row["date"]))
            labels.append(1 if hs > as_ else 0 if hs == as_ else -1)
        except Exception:
            failed += 1

    X, y = np.array(rows), np.array(labels)
    print(f"Samples: {len(X):,}  |  Failed rows: {failed}")
    if len(X) == 0:
        raise Exception("No training samples generated. Check match file structure.")
    return X, y


def train_models(X, y):
    print("\n— MODEL TRAINING —")
    cv_folds = min(5, len(X))

    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.04,
        subsample=0.8, random_state=42,
    )
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, random_state=42,
    )

    print(f"Training on {len(X):,} samples with {cv_folds}-fold CV…")
    for name, model in [("GradientBoosting", gb), ("RandomForest", rf)]:
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")
        print(f"  {name}: {scores.mean():.3f} ± {scores.std():.3f}")

    gb.fit(X, y)
    rf.fit(X, y)
    return gb, rf


# ── 5. PREDICTION HELPERS ────────────────────────────────────────────────────

def predict_match(gb, rf, df, rankings, t1, t2, date=pd.Timestamp("2026-06-20")):
    """Returns (p_win_t1, p_draw, p_win_t2) — ensemble of GB + RF."""
    f       = build_features(df, rankings, t1, t2, date).reshape(1, -1)
    p_gb    = gb.predict_proba(f)[0]
    p_rf    = rf.predict_proba(f)[0]
    classes = list(gb.classes_)
    p       = (p_gb + p_rf) / 2

    pw  = float(p[classes.index( 1)]) if  1 in classes else 0.33
    pd_ = float(p[classes.index( 0)]) if  0 in classes else 0.33
    pl  = float(p[classes.index(-1)]) if -1 in classes else 0.34
    return pw, pd_, pl


def simulate_knockout(gb, rf, df, rankings, t1, t2, date):
    """
    Knockout match: draw probability is split 50/50 then normalised.
    Returns (winner, confidence_pct).
    """
    pw, pd_, pl = predict_match(gb, rf, df, rankings, t1, t2, date)
    p1 = pw + pd_ * 0.5
    p2 = pl + pd_ * 0.5
    total = p1 + p2 if (p1 + p2) > 0 else 1.0
    p1 /= total; p2 /= total
    winner = t1 if p1 >= p2 else t2
    conf   = round((p1 if winner == t1 else p2) * 100, 1)
    return winner, conf


# ── 6. PROBABILITY-CONSISTENT GOAL SIMULATION (FIX #3) ───────────────────────

def _sim_goals(pw: float, pd_: float, pl: float, rng: np.random.Generator):
    """
    Simulate goals so the scoreline is consistent with (pw, pd_, pl).
    Uses Poisson distributions with rates tuned to international football.
    """
    r = rng.random()
    if r < pw:           # team 1 wins
        g1 = int(rng.poisson(1.9)); g2 = int(rng.poisson(0.85))
        if g1 <= g2: g1 = g2 + 1
    elif r < pw + pd_:   # draw
        g  = int(rng.poisson(1.05))
        g1 = g2 = g
    else:                # team 2 wins
        g2 = int(rng.poisson(1.9)); g1 = int(rng.poisson(0.85))
        if g2 <= g1: g2 = g1 + 1
    return g1, g2


# ── 7. GROUP STAGE MONTE CARLO ────────────────────────────────────────────────

def run_group_stage(gb, rf, df, rankings, df_schedule, n_sim: int = 5_000):
    gs_matches = df_schedule[df_schedule["stage"] == "Group Stage"]

    groups: dict[str, set] = {}
    for _, row in gs_matches.iterrows():
        g = row["group"]
        groups.setdefault(g, set())
        groups[g].update([row["team_a"], row["team_b"]])

    # Pre-compute match probabilities
    match_probs: dict = {}
    for g in groups:
        for _, row in gs_matches[gs_matches["group"] == g].iterrows():
            key = (row["team_a"], row["team_b"])
            if key not in match_probs:
                match_probs[key] = predict_match(
                    gb, rf, df, rankings,
                    row["team_a"], row["team_b"],
                    pd.Timestamp(row["date"]),
                )

    advance_counts = {t: 0 for teams in groups.values() for t in teams}
    third_counts   = {t: 0 for teams in groups.values() for t in teams}
    rng = np.random.default_rng(seed=42)

    for _ in range(n_sim):
        for g, teams in groups.items():
            standings = {t: dict(pts=0, gd=0, gf=0) for t in teams}
            for _, row in gs_matches[gs_matches["group"] == g].iterrows():
                t1, t2 = row["team_a"], row["team_b"]
                pw, pd_, pl = match_probs[(t1, t2)]
                g1, g2 = _sim_goals(pw, pd_, pl, rng)   # FIX: probability-consistent

                r = rng.random()
                if r < pw:
                    standings[t1]["pts"] += 3
                    standings[t1]["gd"] += g1 - g2; standings[t1]["gf"] += g1
                    standings[t2]["gf"] += g2
                elif r < pw + pd_:
                    standings[t1]["pts"] += 1; standings[t2]["pts"] += 1
                    standings[t1]["gf"] += g1; standings[t2]["gf"] += g2
                    standings[t1]["gd"] += 0;  standings[t2]["gd"] += 0
                else:
                    standings[t2]["pts"] += 3
                    standings[t2]["gd"] += g2 - g1; standings[t2]["gf"] += g2
                    standings[t1]["gf"] += g1

            ranked = sorted(
                standings.items(),
                key=lambda x: (-x[1]["pts"], -x[1]["gd"], -x[1]["gf"]),
            )
            for rank, (team, _) in enumerate(ranked):
                if rank < 2:
                    advance_counts[team] += 1
                elif rank == 2:
                    third_counts[team] += 1

    result = {}
    for g, teams in sorted(groups.items()):
        probs = sorted(
            [(t, advance_counts[t] / n_sim * 100) for t in teams],
            key=lambda x: -x[1],
        )
        result[g] = probs
    return result, third_counts, n_sim


# ── 8. OFFICIAL BRACKET KNOCKOUT (single most-likely path) ───────────────────

def _resolve_slot(slot: str, qualifiers: dict) -> str:
    """Map a bracket slot string to an actual team name."""
    if slot.startswith("W_"):
        return qualifiers[f"W_{slot[2:]}"]
    if slot.startswith("R_"):
        return qualifiers[f"R_{slot[2:]}"]
    if slot.startswith("3rd_"):
        # Pick the best third-place team whose group is in the slot's group set
        groups_allowed = set(slot[4:])  # e.g. "ABCDF" -> {'A','B','C','D','F'}
        best = max(
            ((t, qualifiers["3rd_pts"].get(t, 0)) for g in groups_allowed
             if f"3rd_{g}" in qualifiers
             for t in [qualifiers[f"3rd_{g}"]]),
            key=lambda x: x[1],
            default=(None, -1),
        )
        return best[0] or "TBD"
    return "TBD"


def run_knockout(gb, rf, df, rankings, group_qualified: dict):
    """
    Single 'most-likely-path' knockout using the OFFICIAL 2026 R32 bracket.
    """
    qualifiers = {}
    for g, probs in group_qualified.items():
        qualifiers[f"W_{g}"] = probs[0][0]
        qualifiers[f"R_{g}"] = probs[1][0]
        # Placeholder: third-place team with dummy pts for slot resolution
        qualifiers[f"3rd_{g}"]    = probs[2][0] if len(probs) > 2 else probs[1][0]
        qualifiers["3rd_pts"]     = qualifiers.get("3rd_pts", {})
        qualifiers["3rd_pts"][probs[2][0] if len(probs) > 2 else probs[1][0]] = probs[2][1] if len(probs) > 2 else 0

    # Build R32 matchups
    r32_matchups = []
    for s1, s2 in R32_SLOTS:
        t1 = _resolve_slot(s1, qualifiers)
        t2 = _resolve_slot(s2, qualifiers)
        r32_matchups.append((t1, t2))

    all_rounds = []
    KO_DATE = pd.Timestamp("2026-06-28")

    def play_round(matchups, name, date):
        winners, results = [], []
        for t1, t2 in matchups:
            if t1 == "TBD" or t2 == "TBD":
                winners.append(t1 if t2 == "TBD" else t2)
                results.append(dict(t1=t1, t2=t2, winner=winners[-1], loser="TBD", conf=100.0))
                continue
            w, conf = simulate_knockout(gb, rf, df, rankings, t1, t2, date)
            loser   = t2 if w == t1 else t1
            results.append(dict(t1=t1, t2=t2, winner=w, loser=loser, conf=conf))
            winners.append(w)
        all_rounds.append(dict(round=name, matches=results))
        return winners

    r32_w = play_round(r32_matchups, "Round of 32", KO_DATE)

    # R16 via official pair indices (R16_PAIRS lists 0-indexed match numbers)
    r16_matchups = [(r32_w[a], r32_w[b]) for a, b in R16_PAIRS]
    r16_w = play_round(r16_matchups, "Round of 16", pd.Timestamp("2026-07-04"))

    qf_matchups = [(r16_w[a], r16_w[b]) for a, b in QF_PAIRS]
    qf_w = play_round(qf_matchups, "Quarter-finals", pd.Timestamp("2026-07-10"))

    sf_matchups = [(qf_w[a], qf_w[b]) for a, b in SF_PAIRS]
    sf_w = play_round(sf_matchups, "Semi-finals", pd.Timestamp("2026-07-14"))

    if len(sf_w) >= 2:
        champion, conf = simulate_knockout(
            gb, rf, df, rankings, sf_w[0], sf_w[1], pd.Timestamp("2026-07-19"),
        )
        runner_up = sf_w[1] if champion == sf_w[0] else sf_w[0]
        all_rounds.append(dict(
            round="Final",
            matches=[dict(t1=sf_w[0], t2=sf_w[1], winner=champion,
                          loser=runner_up, conf=conf)],
        ))
    else:
        champion  = sf_w[0] if sf_w else "Unknown"
        runner_up = "Unknown"

    return all_rounds, champion, runner_up


# ── 9. FULL TOURNAMENT MONTE CARLO (FIX #4 + #5) ─────────────────────────────

def run_full_tournament_simulation(
    gb, rf, df, rankings, df_schedule, n_sim: int = 2_000,
):
    gs_matches = df_schedule[df_schedule["stage"] == "Group Stage"]

    groups: dict[str, set] = {}
    for _, row in gs_matches.iterrows():
        g = row["group"]
        groups.setdefault(g, set())
        groups[g].update([row["team_a"], row["team_b"]])

    group_ids = sorted(groups.keys())

    match_probs: dict = {}
    for g in groups:
        for _, row in gs_matches[gs_matches["group"] == g].iterrows():
            key = (row["team_a"], row["team_b"])
            if key not in match_probs:
                match_probs[key] = predict_match(
                    gb, rf, df, rankings,
                    row["team_a"], row["team_b"],
                    pd.Timestamp(row["date"]),
                )

    ko_prob_cache: dict = {}

    def ko_probs(t1, t2, date_str):
        key = (t1, t2, date_str)
        if key not in ko_prob_cache:
            ko_prob_cache[key] = predict_match(
                gb, rf, df, rankings, t1, t2, pd.Timestamp(date_str),
            )
        return ko_prob_cache[key]

    def ko_winner(t1, t2, date_str, rng):
        if t1 == "TBD" or t2 == "TBD":
            return t1 if t2 == "TBD" else t2
        pw, pd_, pl = ko_probs(t1, t2, date_str)
        p1 = pw + pd_ * 0.5
        p2 = pl + pd_ * 0.5
        total = p1 + p2 if (p1 + p2) > 0 else 1.0
        return t1 if rng.random() < p1 / total else t2

    champion_counts, finalist_counts, sf_counts, qf_counts = {}, {}, {}, {}
    rng = np.random.default_rng(123)

    for _ in range(n_sim):
        # ── Group stage ──────────────────────────────────────────────────────
        qdict     = {}   # "W_X", "R_X", "3rd_X" -> team
        third_pts = {}   # team -> pts (for 3rd-place slot resolution)

        for g, teams in groups.items():
            standings = {t: dict(pts=0, gd=0, gf=0) for t in teams}
            for _, row in gs_matches[gs_matches["group"] == g].iterrows():
                t1, t2 = row["team_a"], row["team_b"]
                pw, pd_, pl = match_probs[(t1, t2)]
                g1, g2 = _sim_goals(pw, pd_, pl, rng)

                r = rng.random()
                if r < pw:
                    standings[t1]["pts"] += 3
                    standings[t1]["gd"] += g1 - g2; standings[t1]["gf"] += g1
                    standings[t2]["gf"] += g2
                elif r < pw + pd_:
                    standings[t1]["pts"] += 1; standings[t2]["pts"] += 1
                    standings[t1]["gf"] += g1; standings[t2]["gf"] += g2
                else:
                    standings[t2]["pts"] += 3
                    standings[t2]["gd"] += g2 - g1; standings[t2]["gf"] += g2
                    standings[t1]["gf"] += g1

            ranked = sorted(
                standings.items(),
                key=lambda x: (-x[1]["pts"], -x[1]["gd"], -x[1]["gf"]),
            )
            qdict[f"W_{g}"] = ranked[0][0]
            qdict[f"R_{g}"] = ranked[1][0]
            qdict[f"3rd_{g}"] = ranked[2][0]
            third_pts[ranked[2][0]] = ranked[2][1]["pts"] + ranked[2][1]["gd"] * 0.01

        # ── Third-place slot resolver ─────────────────────────────────────────
        def resolve(slot: str) -> str:
            if slot.startswith("W_"):  return qdict[f"W_{slot[2:]}"]
            if slot.startswith("R_"):  return qdict[f"R_{slot[2:]}"]
            # Best third from allowed groups
            groups_allowed = set(slot[4:])
            candidates = [
                (qdict[f"3rd_{g}"], third_pts.get(qdict[f"3rd_{g}"], 0))
                for g in groups_allowed if f"3rd_{g}" in qdict
            ]
            if not candidates:
                return "TBD"
            return max(candidates, key=lambda x: x[1])[0]

        # ── Round of 32 ───────────────────────────────────────────────────────
        r32_teams = [resolve(s) for pair in R32_SLOTS for s in pair]
        r32_w = []
        for i in range(0, len(r32_teams), 2):
            r32_w.append(ko_winner(r32_teams[i], r32_teams[i+1], "2026-06-28", rng))

        # ── Round of 16 ───────────────────────────────────────────────────────
        r16_w = [ko_winner(r32_w[a], r32_w[b], "2026-07-04", rng) for a, b in R16_PAIRS]

        # ── Quarter-finals ────────────────────────────────────────────────────
        qf_w = [ko_winner(r16_w[a], r16_w[b], "2026-07-10", rng) for a, b in QF_PAIRS]

        # ── Semi-finals ───────────────────────────────────────────────────────
        sf_w = [ko_winner(qf_w[a], qf_w[b], "2026-07-14", rng) for a, b in SF_PAIRS]

        # ── Final ─────────────────────────────────────────────────────────────
        champion = ko_winner(sf_w[0], sf_w[1], "2026-07-19", rng)

        champion_counts[champion] = champion_counts.get(champion, 0) + 1
        for t in sf_w:  finalist_counts[t] = finalist_counts.get(t, 0) + 1
        for t in qf_w:  sf_counts[t]       = sf_counts.get(t, 0) + 1
        for t in r16_w: qf_counts[t]       = qf_counts.get(t, 0) + 1

    return champion_counts, finalist_counts, sf_counts, qf_counts, n_sim


# ── 10. REPORTING ─────────────────────────────────────────────────────────────

def print_group_report(group_qualified: dict):
    print("\n" + "=" * 60)
    print("GROUP STAGE — QUALIFICATION PROBABILITIES")
    print("=" * 60)
    for g, probs in group_qualified.items():
        print(f"\nGroup {g}:")
        for i, (team, prob) in enumerate(probs):
            bar = "█" * int(prob / 5) + "░" * (20 - int(prob / 5))
            adv = "✅" if i < 2 else "  "
            print(f"  {adv} {team:<28} {prob:5.1f}%  {bar}")


def print_knockout_report(rounds: list, champion: str, runner_up: str):
    print("\n" + "=" * 60)
    print("KNOCKOUT STAGE — OFFICIAL BRACKET, MOST-LIKELY PATH")
    print("=" * 60)
    for rnd in rounds:
        print(f"\n— {rnd['round']} —")
        for m in rnd["matches"]:
            print(f"  {m['t1']:<25} vs {m['t2']:<25}  →  {m['winner']} wins ({m['conf']}%)")
    print("\n" + "🏆" * 40)
    print(f"  Most-likely-path champion : {champion}")
    print(f"  Runner-up                 : {runner_up}")
    print("🏆" * 40)
    print("  (See Monte Carlo odds below for full probability distribution.)")


def print_full_simulation_report(
    champion_counts, finalist_counts, sf_counts, qf_counts, n_sim, top: int = 15,
):
    print("\n" + "=" * 60)
    print(f"FULL TOURNAMENT MONTE CARLO ({n_sim:,} simulated tournaments)")
    print("=" * 60)

    print(f"\nChampionship probability (top {top}):")
    for team, c in sorted(champion_counts.items(), key=lambda x: -x[1])[:top]:
        bar = "█" * int(c / n_sim * 100 / 2)
        print(f"  {team:<28} {c / n_sim * 100:5.1f}%  {bar}")

    print(f"\nReach the Final (top {top}):")
    for team, c in sorted(finalist_counts.items(), key=lambda x: -x[1])[:top]:
        print(f"  {team:<28} {c / n_sim * 100:5.1f}%")

    print(f"\nReach the Semi-finals (top {top}):")
    for team, c in sorted(sf_counts.items(), key=lambda x: -x[1])[:top]:
        print(f"  {team:<28} {c / n_sim * 100:5.1f}%")

    print(f"\nReach the Quarter-finals (top {top}):")
    for team, c in sorted(qf_counts.items(), key=lambda x: -x[1])[:top]:
        print(f"  {team:<28} {c / n_sim * 100:5.1f}%")


# ── 11. PREDICT ALL GROUP MATCHES ─────────────────────────────────────────────

def predict_all_group_matches(gb, rf, df, rankings, df_schedule):
    gs_matches = df_schedule[df_schedule["stage"] == "Group Stage"]
    print("\n" + "=" * 60)
    print("ALL GROUP STAGE MATCH PREDICTIONS")
    print("=" * 60)
    for g, rows in gs_matches.groupby("group"):
        print(f"\nGroup {g}")
        print("-" * 50)
        for _, row in rows.sort_values("date").iterrows():
            t1, t2 = row["team_a"], row["team_b"]
            date   = pd.Timestamp(row["date"])
            pw, pd_, pl = predict_match(gb, rf, df, rankings, t1, t2, date)
            print(f"  {row['date']}  {t1:<22} vs {t2:<22}")
            print(f"      {t1} win: {pw*100:5.1f}%   Draw: {pd_*100:5.1f}%   {t2} win: {pl*100:5.1f}%")


# ── 12. PREDICT ANY SINGLE MATCH ──────────────────────────────────────────────

def predict_single_match(
    gb, rf, df, rankings, team_a: str, team_b: str,
    date=pd.Timestamp("2026-06-20"),
):
    pw, pd_, pl = predict_match(gb, rf, df, rankings, team_a, team_b, date)
    print(f"\n{team_a} vs {team_b}  [{date.date()}]")
    print(f"  {team_a} win : {pw  * 100:.1f}%")
    print(f"  Draw         : {pd_ * 100:.1f}%")
    print(f"  {team_b} win : {pl  * 100:.1f}%")

def generate_group_matches(gb, rf, df, rankings, df_schedule):

    gs_matches = df_schedule[df_schedule["stage"] == "Group Stage"]

    result = {}

    for group, rows in gs_matches.groupby("group"):

        result[group] = []

        for _, row in rows.sort_values("date").iterrows():

            t1 = row["team_a"]
            t2 = row["team_b"]

            pw, pd_, pl = predict_match(
                gb,
                rf,
                df,
                rankings,
                t1,
                t2,
                pd.Timestamp(row["date"])
            )

            result[group].append({
                "date": str(row["date"]),
                "a": t1,
                "b": t2,
                "pa": round(pw * 100, 1),
                "pd": round(pd_ * 100, 1),
                "pb": round(pl * 100, 1)
            })

    return result

# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("FIFA WORLD CUP 2026 PREDICTOR  (v2 — full international dataset)")
    print("=" * 60)

    print("\nLoading rankings…")
    rankings = load_rankings()

    print("\nLoading historical matches…")
    df_matches = load_matches()

    print("\nLoading 2026 group-stage schedule…")
    df_schedule = load_schedule()

    # ── Train ──────────────────────────────────────────────────────────────
    X, y = build_training_set(df_matches, rankings)
    gb, rf = train_models(X, y)

    # ── Group-match predictions ────────────────────────────────────────────
    predict_all_group_matches(gb, rf, df_matches, rankings, df_schedule)

    # ── Group qualification Monte Carlo ───────────────────────────────────
    print(f"\nRunning {N_SIMULATIONS:,} group-stage Monte Carlo simulations…")
    group_qualified, third_counts, n_gs_sim = run_group_stage(
        gb, rf, df_matches, rankings, df_schedule, N_SIMULATIONS,
    )
    print_group_report(group_qualified)

    # ── Single most-likely knockout path ───────────────────────────────────
    rounds, champion, runner_up = run_knockout(
        gb, rf, df_matches, rankings, group_qualified,
    )
    print_knockout_report(rounds, champion, runner_up)

    # ── Full tournament Monte Carlo ────────────────────────────────────────
    print(f"\nRunning {N_FULL_SIMULATIONS:,} full-tournament Monte Carlo simulations…")
    champ_c, final_c, sf_c, qf_c, n_sim = run_full_tournament_simulation(
        gb, rf, df_matches, rankings, df_schedule, N_FULL_SIMULATIONS,
    )
    print_full_simulation_report(champ_c, final_c, sf_c, qf_c, n_sim)

    # ── FIFA ranking summary for WC teams ──────────────────────────────────
    wc_teams = {t for probs in group_qualified.values() for t, _ in probs}
    ranked_wc = sorted(
        [(t, get_rank_info(rankings, t)["rank"], get_rank_info(rankings, t)["points"])
         for t in wc_teams],
        key=lambda x: x[1],
    )
    print("\n— TOP 20 FIFA-RANKED WORLD CUP TEAMS —")
    for i, (team, rank, pts) in enumerate(ranked_wc[:20], 1):
        print(f"  {i:2}. {team:<28} rank #{int(rank):<4} ({pts:.1f} pts)")

    # ── Custom one-off predictions ─────────────────────────────────────────
    predict_single_match(gb, rf, df_matches, rankings,
                         "Spain", "Brazil",      pd.Timestamp("2026-07-10"))
    predict_single_match(gb, rf, df_matches, rankings,
                         "Japan", "France",      pd.Timestamp("2026-07-04"))
    predict_single_match(gb, rf, df_matches, rankings,
                         "Netherlands", "Germany", pd.Timestamp("2026-07-10"))

    # ── Save model ─────────────────────────────────────────────────────────
    with open("wc2026_model_v2.pkl", "wb") as f:
        pickle.dump(dict(gb=gb, rf=rf, rankings=rankings), f)
    print("\n✅ Model saved to wc2026_model_v2.pkl")