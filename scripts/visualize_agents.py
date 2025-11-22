"""
Visualize agent comparison logs (JSONL produced by the evaluation pipeline).

Usage:
    python scripts/visualize_agents.py \
        --input logs/eval_compare_medium.jsonl \
        --outdir logs/plots

Generates several PNG plots in the output directory:
- winners_counts.png        : counts of winners (partial vs debate)
- avg_scores_heatmap.png    : heatmap of average top_scores per agent & stage
- times_boxplot.png         : distribution of time_s for partial/debate
- dominance_vs_f1.png       : scatter of average dominance vs average f1 per agent (by stage)

Requires: pandas, matplotlib, seaborn
"""

import os
import json
import argparse
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            data.append(obj)
    return data


def build_dfs(records):
    rows = []
    scores_rows = []
    for rec in records:
        rid = rec.get('id')
        question = rec.get('question')
        partial = rec.get('partial', {})
        debate = rec.get('debate', {})

        # summary row
        rows.append({
            'id': rid,
            'question': question,
            'partial_winner': partial.get('winner'),
            'partial_time_s': partial.get('time_s'),
            'partial_f1': partial.get('f1'),
            'partial_dominance': partial.get('dominance'),
            'debate_winner': debate.get('winner'),
            'debate_time_s': debate.get('time_s'),
            'debate_f1': debate.get('f1'),
            'debate_dominance': debate.get('dominance'),
        })

        # top_scores long form
        for stage_name, stage in [('partial', partial), ('debate', debate)]:
            top_scores = stage.get('top_scores') or {}
            for agent_name, score in (top_scores.items() if isinstance(top_scores, dict) else []):
                scores_rows.append({
                    'id': rid,
                    'stage': stage_name,
                    'agent': agent_name,
                    'score': score,
                })

    df_entries = pd.DataFrame(rows)
    df_scores = pd.DataFrame(scores_rows)
    return df_entries, df_scores


def plot_winner_counts(df_entries, outdir):
    # count winners for partial & debate
    partial_counts = df_entries['partial_winner'].value_counts(dropna=True)
    debate_counts = df_entries['debate_winner'].value_counts(dropna=True)

    winners = sorted(set(partial_counts.index).union(debate_counts.index))
    df = pd.DataFrame({
        'partial': [partial_counts.get(w, 0) for w in winners],
        'debate': [debate_counts.get(w, 0) for w in winners],
    }, index=winners)

    plt.figure(figsize=(10, max(4, len(winners) * 0.4)))
    df.plot(kind='bar', rot=45)
    plt.title('Winner counts by agent (partial vs debate)')
    plt.tight_layout()
    out = os.path.join(outdir, 'winners_counts.png')
    plt.savefig(out)
    plt.close()
    print('Saved', out)


def plot_avg_scores_heatmap(df_scores, outdir):
    if df_scores.empty:
        print('No top_scores data available for heatmap.')
        return
    pivot = df_scores.pivot_table(index='agent', columns='stage', values='score', aggfunc='mean')
    plt.figure(figsize=(8, max(4, len(pivot) * 0.3)))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis')
    plt.title('Average top_scores per agent and stage')
    plt.tight_layout()
    out = os.path.join(outdir, 'avg_scores_heatmap.png')
    plt.savefig(out)
    plt.close()
    print('Saved', out)


def plot_times_boxplot(df_entries, outdir):
    times = []
    for _, r in df_entries.iterrows():
        times.append({'stage': 'partial', 'time_s': r.get('partial_time_s')})
        times.append({'stage': 'debate', 'time_s': r.get('debate_time_s')})
    tdf = pd.DataFrame(times).dropna()
    if tdf.empty:
        print('No timing data to plot.')
        return
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='stage', y='time_s', data=tdf)
    plt.title('Time (s) distribution by stage')
    plt.tight_layout()
    out = os.path.join(outdir, 'times_boxplot.png')
    plt.savefig(out)
    plt.close()
    print('Saved', out)


def plot_dominance_vs_f1(df_entries, outdir):
    # compute average dominance and f1 per agent and stage (use winner association)
    rows = []
    for _, r in df_entries.iterrows():
        pw = r.get('partial_winner')
        if pw:
            rows.append({'agent': pw, 'stage': 'partial', 'dominance': r.get('partial_dominance'), 'f1': r.get('partial_f1')})
        dw = r.get('debate_winner')
        if dw:
            rows.append({'agent': dw, 'stage': 'debate', 'dominance': r.get('debate_dominance'), 'f1': r.get('debate_f1')})
    if not rows:
        print('No winner dominance/f1 data to plot.')
        return
    rdf = pd.DataFrame(rows).dropna()
    if rdf.empty:
        print('No numeric dominance/f1 data to plot.')
        return
    agg = rdf.groupby(['agent', 'stage']).agg({'dominance': 'mean', 'f1': 'mean'}).reset_index()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=agg, x='dominance', y='f1', hue='agent', style='stage', s=100)
    plt.title('Average dominance vs f1 per agent (by stage)')
    plt.tight_layout()
    out = os.path.join(outdir, 'dominance_vs_f1.png')
    plt.savefig(out)
    plt.close()
    print('Saved', out)


def make_output_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Path to JSONL log file')
    parser.add_argument('--outdir', '-o', default='logs/plots', help='Output directory for plots')
    args = parser.parse_args()

    make_output_dir(args.outdir)

    records = load_jsonl(args.input)
    if not records:
        print('No records found in', args.input)
        return

    df_entries, df_scores = build_dfs(records)

    # Basic CSV exports for quick inspection
    df_entries.to_csv(os.path.join(args.outdir, 'summary_entries.csv'), index=False)
    if not df_scores.empty:
        df_scores.to_csv(os.path.join(args.outdir, 'scores_long.csv'), index=False)

    plot_winner_counts(df_entries, args.outdir)
    plot_avg_scores_heatmap(df_scores, args.outdir)
    plot_times_boxplot(df_entries, args.outdir)
    plot_dominance_vs_f1(df_entries, args.outdir)

    print('All plots saved to', args.outdir)


if __name__ == '__main__':
    main()
