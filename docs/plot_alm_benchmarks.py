"""Render the recorded ALM comparison without rerunning any optimization."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path,
                        default=Path(__file__).with_name('alm_benchmark_results.json'))
    parser.add_argument('--pinned-results', type=Path,
                        help='optional separate benchmark JSON generated with --scipy-pinned-host')
    args = parser.parse_args()
    reports = json.loads(args.results.read_text())
    pairs = {}
    for report in reports:
        key = report['platform'], report['case'], report['dofs']
        backend = 'pinned' if report.get('scipy_pinned_host') else report['backend']
        pairs.setdefault(key, {})[backend] = report
        if report.get('pinned_host_variant'):
            pairs[key]['pinned'] = report['pinned_host_variant']
    if args.pinned_results:
        for report in json.loads(args.pinned_results.read_text()):
            key = report['platform'], report['case'], report['dofs']
            pairs.setdefault(key, {})['pinned'] = report
    pairs = [(key, pair) for key, pair in pairs.items() if {'scipy', 'jaxopt'} <= pair.keys()]
    if not pairs:
        parser.error('Results must include matching SciPy and JAXopt cases')
    plt.rcParams['svg.fonttype'] = 'none'
    platforms = list(dict.fromkeys(key[0] for key, _ in pairs))
    fig, axes = plt.subplots(len(platforms), 3, figsize=(13, 3.5 * len(platforms) + 1.2), squeeze=False)
    styles = [('scipy', 'SciPy', '#2378a8'),
              ('pinned', 'SciPy: pinned host outputs (GPU)', '#20916e'),
              ('jaxopt', 'JAXopt', '#a45265')]
    metrics = [('warm_median_s', 1000., 'Warm update', 'Time (ms)'),
               ('first_update_s', 1., 'First update, including compilation', 'Time (s, log scale)'),
               ('peak_rss_mib', 1 / 1024, 'Peak process host RAM', 'RAM (GiB)')]
    for row, platform in enumerate(platforms):
        cases = [(key, pair) for key, pair in pairs if key[0] == platform]
        labels = [f'{"Quadratic" if case == "quadratic" else case.upper() + " smoke"}\n{size} DOFs'
                  for (_, case, size), _ in cases]
        variants = [style for style in styles if any(style[0] in pair for _, pair in cases)]
        width = .76 / len(variants)
        for column, (metric, scale, title, unit) in enumerate(metrics):
            ax = axes[row, column]
            extent = []
            for offset, (backend, _, color) in enumerate(variants):
                available = [(index, pair[backend]) for index, (_, pair) in enumerate(cases) if backend in pair]
                positions = np.array([index for index, _ in available]) + (offset - (len(variants) - 1) / 2) * width
                values = np.array([report[metric] * scale for _, report in available])
                lower, upper = values.copy(), values.copy()
                if metric == 'warm_median_s':
                    lower = np.array([min(report['warm_update_s']) * scale for _, report in available])
                    upper = np.array([max(report['warm_update_s']) * scale for _, report in available])
                ax.bar(positions, values, width * .88, color=color,
                       yerr=np.array([values - lower, upper - values]) if metric == 'warm_median_s' else None,
                       error_kw=dict(capsize=2, linewidth=.8, alpha=.65))
                for position, value, top in zip(positions, values, upper):
                    label = f'{value:.1f}' if metric != 'peak_rss_mib' and value >= 10 else f'{value:.2f}'
                    ax.annotate(label, (position, top), xytext=(0, 4),
                                textcoords='offset points', ha='center', va='bottom', fontsize=8)
                extent.extend(upper)
            if metric == 'first_update_s':
                ax.set_yscale('log')
                ax.set_ylim(min(extent) * .6, max(extent) * 2.1)
            else:
                ax.set_ylim(0, max(extent) * 1.23)
            ax.set_xticks(range(len(cases)), labels, fontsize=9)
            ax.set_title(f'{"GPU" if platform == "cuda" else platform.upper()} | {title}', fontsize=10)
            ax.set_ylabel(unit, fontsize=9)
            ax.grid(axis='y', alpha=.18)
            ax.set_axisbelow(True)
    fig.suptitle('ALM comparison with SciPy pinned host outputs', fontsize=15, y=.985)
    fig.legend(handles=[Patch(color=color, label=label) for backend, label, color in styles
                        if any(backend in pair for _, pair in pairs)],
               loc='upper center', bbox_to_anchor=(.5, .95), ncol=3, frameon=False, fontsize=10)
    fig.text(.5, .035, 'Float64 | AMD Ryzen 7 250 CPU / NVIDIA RTX 5060 Laptop GPU | Lower is better',
             ha='center', fontsize=9)
    fig.text(.5, .012, 'Warm bars: median of 3 fixed-state repeats; whiskers: min/max. QS/QI smoke runs are unconverged.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .90), h_pad=2.3, w_pad=1.7)
    for extension in ('svg', 'png'):
        fig.savefig(args.results.with_name(f'alm_benchmarks.{extension}'), dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
