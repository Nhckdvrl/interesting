"""Multi-host GPU scheduler.

IMPORTANT: matched comparisons must not be split across GPU architectures.
We are resolving effects of order 1e-4 nats; A100 (sm80) and RTX PRO 6000
(sm120) do not produce bitwise-identical bf16/fp32 reductions.  Every panel
therefore declares an `arch` and only gets slots of that architecture.
"""
import argparse, json, os, queue, subprocess, threading, time

REPO = "/home/xiang/interesting"
PY = {"a100": f"{REPO}/.venv-a100/bin/python",
      "blackwell": f"{REPO}/.venv/bin/python"}

# host -> (arch, [gpu ids we may use])
POOL = {
    "fvcrc10": ("a100", [0, 1, 2, 3]),
    "fvcrc11": ("a100", [0, 1, 2]),
    "fvcrc12": ("a100", [0, 1]),
    "fvcrc15": ("a100", [2, 3]),
    "fvcrc20": ("blackwell", [0, 1, 2, 3]),
    "LOCAL":   ("blackwell", [0, 1, 2, 3]),
}

MEM_FREE_MIN = 60000   # MiB required free to claim a slot


def probe(hosts=None):
    """Return list of (host, gpu, arch) slots that are currently free enough."""
    out = []
    for host, (arch, gpus) in POOL.items():
        if hosts and host not in hosts:
            continue
        cmd = "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
        try:
            if host == "LOCAL":
                r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                   timeout=60)
            else:
                r = subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd],
                                   capture_output=True, text=True, timeout=90)
            if r.returncode != 0:
                print(f"  {host}: unreachable"); continue
        except Exception as e:
            print(f"  {host}: {e}"); continue
        for line in r.stdout.strip().split("\n"):
            i, used, total, util = [int(x.strip()) for x in line.split(",")]
            if i in gpus and (total - used) >= MEM_FREE_MIN:
                out.append((host, i, arch))
    return out


def run(jobs, slots, logdir, py_key=None, dry=False, env_extra=""):
    """jobs: list of command strings containing the literal token {PY}."""
    os.makedirs(logdir, exist_ok=True)
    q = queue.Queue()
    for i, j in enumerate(jobs):
        q.put((i, j))
    lock = threading.Lock()
    fails, done = [], [0]
    t_start = time.time()

    def worker(host, gpu, arch):
        py = PY[py_key or arch]
        while True:
            try:
                i, cmd = q.get_nowait()
            except queue.Empty:
                return
            full = cmd.replace("{PY}", py)
            lf = os.path.join(logdir, f"job{i:04d}.log")
            # PYTHONDONTWRITEBYTECODE: the repo lives on NFS and stale .pyc
            # files from a different node's clock have silently shadowed source
            # edits before.
            sh = (f"cd {REPO} && CUDA_VISIBLE_DEVICES={gpu} "
                  f"HF_HUB_OFFLINE=0 PYTHONDONTWRITEBYTECODE=1 "
                  f"{env_extra} {full}")
            if dry:
                print(f"[{host}:{gpu}] {sh}"); continue
            t0 = time.time()
            with open(lf, "w") as f:
                f.write(f"# {host}:{gpu} ({arch})\n{sh}\n\n"); f.flush()
                if host == "LOCAL":
                    rc = subprocess.call(sh, shell=True, stdout=f,
                                         stderr=subprocess.STDOUT)
                else:
                    rc = subprocess.call(
                        ["ssh", "-o", "BatchMode=yes", host, sh],
                        stdout=f, stderr=subprocess.STDOUT)
            with lock:
                done[0] += 1
                tail = ""
                try:
                    tail = [l for l in open(lf).read().strip().split("\n")
                            if l.strip()][-1][:150]
                except Exception:
                    pass
                el = time.time() - t_start
                print(f"[{host}:{gpu}] {done[0]}/{len(jobs)} job{i:04d} rc={rc} "
                      f"{time.time()-t0:.0f}s (elapsed {el/60:.1f}m)  {tail}",
                      flush=True)
                if rc != 0:
                    fails.append((i, sh, lf))

    ths = [threading.Thread(target=worker, args=s) for s in slots]
    for t in ths: t.start()
    for t in ths: t.join()
    if fails:
        print(f"\n{len(fails)} FAILED")
        for i, c, lf in fails[:12]:
            print("  ", lf)
    return fails


def main(jobs, tag, logdir, arch="a100", hosts=None, dry=False):
    slots = [s for s in probe(hosts) if s[2] == arch]
    print(f"{len(jobs)} jobs, {len(slots)} free {arch} slots: "
          + ", ".join(f"{h}:{g}" for h, g, _ in slots))
    if not slots:
        raise SystemExit("no free slots")
    return run(jobs, slots, logdir, dry=dry)


if __name__ == "__main__":
    print("free slots:")
    for h, g, a in probe():
        print(f"  {h}:{g} ({a})")
