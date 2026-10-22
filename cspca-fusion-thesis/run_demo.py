"""One-command demo of the analysis pipeline on SYNTHETIC data.

It proves that the code runs end to end on your machine. The numbers are meaningless clinically and will NOT match the thesis.
To reproduce the thesis numbers you need the real public datasets (see RUN_GUIDE.md).

Usage:
    python run_demo.py            # all steps, about 7 minutes
    python run_demo.py --quick    # skips the two extra model configurations (about 3 minutes)
Outputs: demo_output/demo_log.txt (everything printed by every step) and demo_output/logs/<step>.txt
"""
import os, subprocess, sys, time, pathlib, argparse

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--tail", type=int, default=12, help="lines of each step's output to show on screen")
a = ap.parse_args()
root = pathlib.Path(__file__).resolve().parent; analysis = root / "analysis"; demo = root / "demo_output"; logs = demo / "logs"
logs.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, CSPCA_ROOT=str(demo / "synthetic"), CSPCA_FAST="1", PYTHONIOENCODING="utf-8")

steps = [("make_synthetic_data", ["make_synthetic_data.py"]),
         ("01_clinical_table", ["01_build_picai_clinical_table.py"]),
         ("03_fused_dataset", ["03_build_fused_dataset.py", "--n-components", "20"]),     # the synthetic data has 64 features
         ("06_train_config_A_fully_automated", ["06_train_algorithms.py", "--config", "A"])]
if not a.quick:
    steps += [("06_train_config_B_pirads_informed", ["06_train_algorithms.py", "--config", "B"]),
              ("06_train_config_C_biopsy_pathway", ["06_train_algorithms.py", "--config", "C"])]
steps += [("07_locked_model_external", ["07_locked_model_external.py"]),
          ("08_leakage_audit", ["08_leakage_audit.py"]),
          ("09_leave_one_centre_out", ["09_leave_one_centre_out.py"]),
          ("10_baselines_statistics", ["10_baselines_statistics.py"]),
          ("11_external_investigation", ["11_external_investigation.py"]),
          ("12_pca_sensitivity", ["12_pca_sensitivity.py"])]

summary = []; t_all = time.time()
with open(demo / "demo_log.txt", "w", encoding="utf-8") as full:
    for name, cmd in steps:
        print(f"\n=== {name}", flush=True); t0 = time.time()
        r = subprocess.run([sys.executable] + cmd, cwd=analysis, env=env, capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or ""); dt = time.time() - t0
        (logs / f"{name}.txt").write_text(out, encoding="utf-8")
        full.write(f"\n===== {name} ({'OK' if r.returncode == 0 else 'FAILED'}, {dt:.0f}s) =====\n{out}\n")
        lines = [l for l in out.strip().splitlines() if l.strip()]
        print("\n".join("    " + l[:160] for l in lines[-a.tail:]))
        print(f"    -> {'OK' if r.returncode == 0 else 'FAILED'} ({dt:.0f}s)", flush=True)
        summary.append((name, r.returncode == 0, dt))
        if r.returncode != 0: break

print("\n" + "=" * 60 + "\nSUMMARY (synthetic data: results are not clinically meaningful)")
for name, ok, dt in summary: print(f"  {'OK    ' if ok else 'FAILED'} {name:42s} {dt:6.0f}s")
print(f"Total: {time.time() - t_all:.0f}s | full log: {demo / 'demo_log.txt'}")
sys.exit(0 if all(ok for _, ok, _ in summary) and len(summary) == len(steps) else 1)
