from flask import Flask, render_template, jsonify
import pickle
import WC2026

app = Flask(__name__)

print("Loading model...")

with open("wc2026_model_v2.pkl", "rb") as f:
    saved = pickle.load(f)

gb = saved["gb"]
rf = saved["rf"]
rankings = saved["rankings"]

print("Loading datasets...")

df_matches = WC2026.load_matches()
df_schedule = WC2026.load_schedule()

print("Generating dashboard data (one-time)...")

group_qualified, _, _ = WC2026.run_group_stage(
    gb,
    rf,
    df_matches,
    rankings,
    df_schedule,
    100      # use 100 first for testing
)

rounds, champion, runner_up = WC2026.run_knockout(
    gb,
    rf,
    df_matches,
    rankings,
    group_qualified
)

champ_c, final_c, sf_c, qf_c, _ = \
    WC2026.run_full_tournament_simulation(
        gb,
        rf,
        df_matches,
        rankings,
        df_schedule,
        100      # use 100 first for testing
    )
group_matches = WC2026.generate_group_matches(
    gb,
    rf,
    df_matches,
    rankings,
    df_schedule
)
DASHBOARD_DATA = {
    "groups": group_qualified,
    "group_matches": group_matches,
    "rounds": rounds,
    "champion": champion,
    "runner_up": runner_up,
    "championship_odds": champ_c,
    "final_odds": final_c,
    "sf_odds": sf_c,
    "qf_odds": qf_c
}

print("Dashboard ready!")


@app.route("/")
def home():
    return render_template("wc2026.html")


@app.route("/api/dashboard")
def dashboard():
    return jsonify(DASHBOARD_DATA)


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)