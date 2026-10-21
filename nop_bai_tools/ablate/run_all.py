import subprocess, sys, os, time
from concurrent.futures import ThreadPoolExecutor
R = "scratch/robust2"; O = "scratch/ablate"
JOBS = {"no1e": f"{R}/base.mp4", "ema65": f"{R}/base.mp4", "nolk": f"{R}/base.mp4",
        "bright_enh": f"{R}/bright.mp4", "base_enh": f"{R}/base.mp4"}
def run(name):
    csvp, txt = f"{O}/{name}.csv", f"{O}/run_{name}.txt"
    t = time.time()
    with open(txt, "w", encoding="utf-8") as fo:
        fo.write(f"### {name}\n"); fo.flush()
        subprocess.run([sys.executable, f"{O}/child.py", name, JOBS[name], csvp], stdout=fo, stderr=subprocess.STDOUT,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return name, round(time.time() - t)
with ThreadPoolExecutor(5) as ex:
    for r in ex.map(run, list(JOBS)):
        print("done", r, flush=True)
open(f"{O}/DONE", "w").write("ok")
