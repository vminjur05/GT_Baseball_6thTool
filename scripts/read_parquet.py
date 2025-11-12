"""
Improved Parquet inspector and visualizer for GT Baseball 6th Tool.
- Reads parquet metadata quickly
- Samples large files without full load
- Produces numeric/categorical summaries
- Saves simple accountability score + per-player summary (if fields available)
- Writes a few PNG charts to an output folder
"""
import argparse
from pathlib import Path
import json

import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def read_parquet_sample(path: Path, sample_size: int = 5000):
    pf = pq.ParquetFile(str(path))
    num_rows = pf.metadata.num_rows
    schema = pf.schema_arrow
    if num_rows <= sample_size:
        table = pf.read()
        df = table.to_pandas()
        full = True
    else:
        # read row-groups until we have enough rows (fast, avoids full memory spike)
        parts = []
        rows = 0
        for rg in range(pf.num_row_groups):
            rg_tbl = pf.read_row_group(rg)
            parts.append(rg_tbl.to_pandas())
            rows += rg_tbl.num_rows
            if rows >= sample_size:
                break
        df = pd.concat(parts, ignore_index=True)
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=42).reset_index(drop=True)
        full = False
    return df, schema, num_rows, full


def summarize_df(df: pd.DataFrame):
    info = {}
    info["rows"] = int(len(df))
    info["columns"] = list(df.columns)
    info["dtypes"] = {c: str(df[c].dtype) for c in df.columns}
    info["memory_mb"] = df.memory_usage(deep=True).sum() / 1024 ** 2
    # numeric summary
    numeric = df.select_dtypes("number")
    info["numeric_desc"] = numeric.describe().to_dict()
    # top categories for non-numeric
    cats = {}
    for c in df.select_dtypes(include=["object", "category"]).columns:
        cats[c] = df[c].value_counts(dropna=False).head(10).to_dict()
    info["top_categories"] = cats
    return info


def compute_simple_accountability(df: pd.DataFrame):
    # requires: reaction_time, route_efficiency, Probability, pop_time/throw_velo
    out = {}
    # normalization helpers
    def norm_inv(s):
        s = s.dropna()
        if s.empty:
            return pd.Series(dtype=float)
        lo, hi = np.nanpercentile(s, [5, 95])
        scaled = 1 - (np.clip(s, lo, hi) - lo) / max(1e-6, hi - lo)
        return scaled

    def norm_pos(s):
        s = s.dropna()
        if s.empty:
            return pd.Series(dtype=float)
        lo, hi = np.nanpercentile(s, [5, 95])
        scaled = (np.clip(s, lo, hi) - lo) / max(1e-6, hi - lo)
        return scaled

    df_scores = pd.DataFrame(index=df.index)
    weights = {"reaction": 0.35, "route": 0.35, "prob": 0.2, "throw": 0.1}

    if "reaction_time" in df.columns:
        df_scores["s_reaction"] = norm_inv(df["reaction_time"]).reindex(df.index).fillna(0.5)
    else:
        df_scores["s_reaction"] = 0.5

    if "route_efficiency" in df.columns:
        df_scores["s_route"] = norm_pos(df["route_efficiency"]).reindex(df.index).fillna(0.5)
    else:
        df_scores["s_route"] = 0.5

    if "Probability" in df.columns:
        df_scores["s_prob"] = norm_pos(df["Probability"]).reindex(df.index).fillna(0.5)
    else:
        df_scores["s_prob"] = 0.5

    if "pop_time" in df.columns and df["pop_time"].notna().any():
        df_scores["s_throw"] = norm_inv(df["pop_time"]).reindex(df.index).fillna(0.5)
    elif "throw_velo" in df.columns and df["throw_velo"].notna().any():
        df_scores["s_throw"] = norm_pos(df["throw_velo"]).reindex(df.index).fillna(0.5)
    else:
        df_scores["s_throw"] = 0.5

    df["accountability_score"] = (
        weights["reaction"] * df_scores["s_reaction"]
        + weights["route"] * df_scores["s_route"]
        + weights["prob"] * df_scores["s_prob"]
        + weights["throw"] * df_scores["s_throw"]
    )

    # simple flag: all three core criteria pass with default thresholds
    df["accountable_flag"] = False
    conds = []
    if "reaction_time" in df.columns:
        conds.append(df["reaction_time"] <= 1.2)
    if "route_efficiency" in df.columns:
        conds.append(df["route_efficiency"] >= 80)
    if "Probability" in df.columns:
        # Probability might be 0-1 or 0-100
        prob = df["Probability"].dropna()
        if prob.max() <= 1.1:
            conds.append(df["Probability"] >= 0.6)
        else:
            conds.append(df["Probability"] >= 60)
    if conds:
        df["accountable_flag"] = np.logical_and.reduce(conds)

    # per-player summary (use 'name' or 'player_id' if present)
    player_col = None
    if "name" in df.columns:
        player_col = "name"
    elif "player_id" in df.columns:
        player_col = "player_id"

    if player_col:
        summary = (
            df.groupby(player_col)
            .agg(
                plays=("ffx_play_guid", "nunique")
                if "ffx_play_guid" in df.columns
                else ("accountability_score", "count"),
                pct_accountable=("accountable_flag", lambda x: 100 * x.sum() / max(1, len(x))),
                median_score=("accountability_score", "median"),
                median_reaction=("reaction_time", "median"),
                median_route=("route_efficiency", "median"),
            )
            .reset_index()
            .sort_values("median_score", ascending=False)
        )
    else:
        summary = None

    out["df_with_scores"] = df
    out["player_summary"] = summary
    return out


def quick_plots(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    plots = []
    def savefig(fig, name):
        p = outdir / name
        fig.savefig(p, bbox_inches="tight", dpi=150)
        plt.close(fig)
        plots.append(str(p))

    # reaction_time distribution
    if "reaction_time" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.histplot(df["reaction_time"].dropna(), bins=30, kde=True, ax=ax)
        ax.set_title("Reaction Time Distribution")
        ax.set_xlabel("Reaction Time (s)")
        savefig(fig, "reaction_time_hist.png")

    # route_efficiency
    if "route_efficiency" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.histplot(df["route_efficiency"].dropna(), bins=30, kde=True, ax=ax)
        ax.set_title("Route Efficiency")
        savefig(fig, "route_efficiency_hist.png")

    # Probability / catch probability
    if "Probability" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.histplot(df["Probability"].dropna(), bins=30, kde=True, ax=ax)
        ax.set_title("Probability (modelled catch chance)")
        savefig(fig, "probability_hist.png")

    # scatter reaction vs route colored by position
    if "reaction_time" in df.columns and "route_efficiency" in df.columns:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        hue = df["pos"] if "pos" in df.columns else None
        sns.scatterplot(
            data=df.sample(min(len(df), 2000), random_state=1),
            x="reaction_time",
            y="route_efficiency",
            hue=hue,
            alpha=0.7,
            ax=ax,
            edgecolor=None,
            s=30,
        )
        ax.set_title("Reaction Time vs Route Efficiency")
        savefig(fig, "reaction_vs_route.png")

    # accountability score histogram
    if "accountability_score" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.histplot(df["accountability_score"].dropna(), bins=30, kde=True, ax=ax)
        ax.set_title("Accountability Score Distribution")
        savefig(fig, "accountability_score_hist.png")

    return plots


def main():
    p = argparse.ArgumentParser(description="Inspect and visualize GT 6th-tool parquet")
    p.add_argument("parquet", help="path to parquet file")
    p.add_argument("--sample", type=int, default=5000, help="rows to sample for preview")
    p.add_argument("--outdir", default="reports/parquet_inspect", help="output folder for plots and summaries")
    args = p.parse_args()

    path = Path(args.parquet)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Reading metadata for {path} ...")
    pf = pq.ParquetFile(str(path))
    print("Num row groups:", pf.num_row_groups)
    print("Total rows:", pf.metadata.num_rows)
    print("Schema:")
    print(pf.schema_arrow)

    print(f"\nLoading sample (up to {args.sample} rows)...")
    df, schema, total_rows, full = read_parquet_sample(path, sample_size=args.sample)
    print(f"Loaded rows: {len(df)}  (full_load={full})")

    info = summarize_df(df)
    # write summary json and csv (top-level)
    (outdir / "summary.json").write_text(json.dumps(info, default=lambda o: str(o), indent=2))
    pd.DataFrame.from_dict(info["numeric_desc"], orient="index").to_csv(outdir / "numeric_description.csv")

    print("\nTop-level info:")
    print(f"Rows (sample): {info['rows']}")
    print(f"Cols: {len(info['columns'])}")
    print(f"Memory (MB): {info['memory_mb']:.2f}")

    # compute accountability-like metrics if possible
    acc = compute_simple_accountability(df)
    df_with_scores = acc["df_with_scores"]
    player_summary = acc["player_summary"]
    if player_summary is not None:
        player_summary.to_csv(outdir / "player_accountability_summary.csv", index=False)
        print("\nPer-player accountability summary written to:", outdir / "player_accountability_summary.csv")
        print(player_summary.head(10).to_string(index=False))
    else:
        print("\nNo player column found; skipping per-player summary.")

    # save scored sample rows
    df_with_scores.head(500).to_csv(outdir / "sample_with_scores.csv", index=False)

    # create plots
    print("\nGenerating plots ...")
    plots = quick_plots(df_with_scores, outdir)
    print("Saved plots:", plots)

    print(f"\nAll outputs written to {outdir}")


if __name__ == "__main__":
    main()