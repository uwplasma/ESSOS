"""Run the ESSOS example with patched inputs, writing figures sized for a JPP page.

python run.py OUTPUT_DIR 'NAME = value' ... '+statement' ...

'NAME = value' replaces the first assignment of NAME in the example. '+statement' lines are
inserted right after the VMEX settings are completed (e.g. '+VMEX.update(ns=(17, 33))').
Figures are rescaled to 6.6 in wide, suptitles dropped and 3D axes hidden.
"""
import os
import re
import sys

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.abspath(sys.argv[1])
sys.path.insert(0, D)
os.chdir(D)

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt

import nearaxis_finite_beta_helpers as helpers

TARGET = 6.6
_subplots, _figure = plt.subplots, plt.figure


def _rescale(kwargs):
    if "figsize" in kwargs:
        width, height = kwargs["figsize"]
        kwargs["figsize"] = (TARGET, height * TARGET / width)
    return kwargs


plt.subplots = lambda *a, **k: _subplots(*a, **_rescale(k))
plt.figure = lambda *a, **k: _figure(*a, **_rescale(k))
matplotlib.figure.Figure.suptitle = lambda self, *a, **k: None
_equal = helpers._equal_3d


def _equal_without_axes(axes, clouds):
    _equal(axes, clouds)
    for ax in axes:
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 1), zoom=1.75)


helpers._equal_3d = _equal_without_axes

path = f"{D}/optimize_coils_and_nearaxis_finite_beta.py"
src = open(path).read()
edits = [a for a in sys.argv[2:] if not a.startswith("+")] + [f'OUTPUT_DIR = Path("{out}")', "SHOW_PLOTS = False"]
extra = [a[1:] for a in sys.argv[2:] if a.startswith("+")]
head_end = src.index('"""', src.index('"""') + 3) + 3  # Never edit the module docstring.
head, src = src[:head_end], src[head_end:]
for edit in edits:
    name = edit.split("=")[0].strip()
    src, count = re.subn(rf"(^|; ){re.escape(name)} = [^\n;#]*", lambda m: m.group(1) + edit + "  ", src,
                         count=1, flags=re.M)
    assert count == 1, f"constant {name} not found"
src = head + src
anchor = 'VMEX.update(case.get("vmex", {}))\n'
assert src.count(anchor) == 1
src = src.replace(anchor, anchor + "".join(line + "\n" for line in extra))
exec(compile(src, path, "exec"), {"__file__": path, "__name__": "__main__"})
