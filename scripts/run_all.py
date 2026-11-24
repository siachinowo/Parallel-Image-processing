import subprocess, os, sys
EXE = "build/bench"
SIZES   = [256, 512, 1024, 2048, 4096]
BATCHES = [1, 10, 50, 100]
CSV = "results/results.csv"
small = len(sys.argv) > 1 and sys.argv[1] == "small"
if small: SIZES = [256, 512]

if os.path.exists(CSV): os.remove(CSV)
for s in SIZES:
    for b in BATCHES:
        trials = 3 if small else (10 if s <= 1024 else 5)
        cmd = [EXE, "--dir", f"data/{s}", "--batch", str(b),
               "--trials", str(trials), "--warmup", "2", "--csv", CSV]
        if b == 1: cmd += ["--save", "results/samples"]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
