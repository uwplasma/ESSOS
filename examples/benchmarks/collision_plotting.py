"""Shared presentation style for collision benchmark figures."""
from functools import wraps
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def presentation(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with plt.rc_context({
            'font.size': 16, 'font.weight': 'bold', 'axes.labelweight': 'bold',
            'axes.titleweight': 'bold', 'axes.titlesize': 18, 'axes.labelsize': 17,
            'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 11,
            'axes.linewidth': 1.2, 'lines.linewidth': 2.6, 'figure.facecolor': 'white',
            'savefig.facecolor': 'white', 'pdf.fonttype': 42, 'mathtext.fontset': 'dejavusans',
        }):
            return function(*args, **kwargs)
    return wrapped


def series(ax, times, values, half=None, *, std=False, scale=1., reference=False, label=None, color='C0'):
    values = np.asarray(values, float)
    half = np.zeros_like(values) if half is None else np.asarray(half)
    low, high = values-half, values+half
    if std:
        values, low, high = (np.sqrt(np.maximum(x, 0.)) for x in (values, low, high))
    values, low, high = (x/scale for x in (values, low, high))
    color = 'black' if reference else color
    if np.any(half):
        ax.fill_between(times, low, high, color=color, alpha=.12 if reference else .18, linewidth=0)
    ax.plot(times, values, color=color, label=label, zorder=8 if reference else 4,
            marker=None if reference or len(times)>16 else 'o', markersize=4)
    ax.grid(False)
    ax.tick_params(direction='in', length=5)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4), useOffset=False)
    ax.set_xlim(times[0], times[-1])


def histogram(ax, edges, counts, density, *, particles, initial=None, label='Simulation', curve=None):
    edges, counts = np.asarray(edges), np.asarray(counts)
    centers, widths = (edges[1:]+edges[:-1])/2, np.diff(edges)
    ax.bar(centers, counts, width=widths*.94, color='tab:orange', alpha=.75,
           edgecolor='.25', linewidth=.7, label=label, zorder=2)
    x, y = (centers, np.asarray(density)*particles*widths) if curve is None else (curve[0], curve[1]*particles*widths[0])
    ax.plot(x, y, color='black', lw=2.8, label='Independent theory', zorder=10)
    if initial is not None:
        ax.axvline(initial, color='blue', ls='--', lw=2.4, label='Initial value', zorder=9)
    ax.set_ylabel('Counts')
    ax.set_ylim(bottom=0)
    ax.grid(False)
    ax.tick_params(direction='in', length=5)


def save(figure, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix('.png'), dpi=320, bbox_inches='tight')
    figure.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(figure)
