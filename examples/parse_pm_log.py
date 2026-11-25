#!/usr/bin/env python3
import re, sys
from pathlib import Path
import pandas as pd

STAGE1_RE = re.compile(
    r"\[Stage 1\]\s+converged=(?P<converged>True|False)\s+"
    r"nit=(?P<nit>\d+)\s+loss=(?P<loss>[\d.eE+-]+)\s+message=(?P<message>.*)"
)
ANNEAL_RE = re.compile(
    r"\[Anneal\s*(?P<stage>\d+)/(?P<total>\d+)\]\s+"
    r"wD=(?P<wD>[\d.eE+-]+)\s+loss=(?P<loss>[\d.eE+-]+)\s+"
    r"nit=(?P<nit>\d+)\s+fB=(?P<fB>[\d.eE+-]+)\s+fV=(?P<fV>[\d.eE+-]+)\s+"
    r"fD=(?P<fD>[\d.eE+-]+)\s+converged=(?P<converged>True|False)"
)
RESTART_IMPROVED_RE = re.compile(
    r"\[(?P<label>\w+)\s+restart\s+(?P<restart>\d+)\]\s+jump IMPROVED loss -> "
    r"(?P<loss>[\d.eE+-]+)"
)
RESTART_NOIMPROVE_RE = re.compile(
    r"\[(?P<label>\w+)\s+restart\s+(?P<restart>\d+)\]\s+jump did not improve\s+"
    r"\(loss=(?P<loss>[\d.eE+-]+)\s+vs best=(?P<best>[\d.eE+-]+)\)"
)
DONE_RE = re.compile(r"Done in\s+(?P<seconds>\d+)s")
FINAL_SMOOTH_RE = re.compile(r"Final \(smooth\):\s+(?P<fB>[\d.eE+-]+)")
HARD_ROUNDED_RE = re.compile(
    r"Hard-rounded:\s+(?P<fB>[\d.eE+-]+)\s+fV=(?P<fV_unique>[\d.eE+-]+)\s+"
    r"cm3 \(unique\)\s+=\s+(?P<fV_full>[\d.eE+-]+)\s+cm3 \(full\)"
)
ACTIVE_MAGNETS_RE = re.compile(
    r"Active magnets:\s+(?P<n_active>\d+)\s*/\s*(?P<n_total>\d+)\s+"
    r"\((?P<pct>[\d.]+)%\)"
)

def parse_log(text):
    stage1_row = None
    anneal_rows = []
    restart_rows = []
    summary = {}
    for line in text.splitlines():
        if m := STAGE1_RE.search(line):
            stage1_row = {"converged": m["converged"] == "True", "nit": int(m["nit"]),
                          "loss": float(m["loss"]), "message": m["message"].strip()}
        elif m := ANNEAL_RE.search(line):
            anneal_rows.append({"stage": int(m["stage"]), "total_stages": int(m["total"]),
                                "wD": float(m["wD"]), "loss": float(m["loss"]), "nit": int(m["nit"]),
                                "fB": float(m["fB"]), "fV": float(m["fV"]), "fD": float(m["fD"]),
                                "converged": m["converged"] == "True"})
        elif m := RESTART_IMPROVED_RE.search(line):
            restart_rows.append({"label": m["label"], "restart": int(m["restart"]),
                                 "loss": float(m["loss"]), "improved": True, "best_so_far": float(m["loss"])})
        elif m := RESTART_NOIMPROVE_RE.search(line):
            restart_rows.append({"label": m["label"], "restart": int(m["restart"]),
                                 "loss": float(m["loss"]), "improved": False, "best_so_far": float(m["best"])})
        elif m := DONE_RE.search(line):
            summary["wall_time_s"] = int(m["seconds"])
        elif m := FINAL_SMOOTH_RE.search(line):
            summary["fB_final_smooth"] = float(m["fB"])
        elif m := HARD_ROUNDED_RE.search(line):
            summary["fB_hard_rounded"] = float(m["fB"])
            summary["fV_unique_cm3"] = float(m["fV_unique"])
            summary["fV_full_cm3"] = float(m["fV_full"])
        elif m := ACTIVE_MAGNETS_RE.search(line):
            summary["n_active"] = int(m["n_active"])
            summary["n_total"] = int(m["n_total"])
            summary["active_pct"] = float(m["pct"])
    return {"stage1": stage1_row, "anneal": pd.DataFrame(anneal_rows),
            "restarts": pd.DataFrame(restart_rows), "summary": summary}

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python parse_pm_log.py <logfile>")
    log_path = Path(sys.argv[1])
    text = log_path.read_text(encoding="utf-8")
    result = parse_log(text)
    print("=== Stage 1 ===")
    print(result["stage1"])
    print()
    print("=== Anneal sub-stages ===")
    print(result["anneal"].to_string(index=False))
    print()
    print("=== Restart attempts ===")
    print(result["restarts"].to_string(index=False) if not result["restarts"].empty else "(none)")
    print()
    print("=== Final summary ===")
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")
    out_dir = log_path.parent
    result["anneal"].to_csv(out_dir / "anneal_stages.csv", index=False)
    result["restarts"].to_csv(out_dir / "restarts.csv", index=False)
    print(f"\nSaved anneal_stages.csv and restarts.csv to {out_dir}")

if __name__ == "__main__":
    main()
