"""Tiny GPU job scheduler: run a list of shell commands across N GPUs."""
import argparse, json, os, subprocess, sys, threading, queue, time


def run_jobs(jobs, gpus, logdir, dry=False):
    os.makedirs(logdir, exist_ok=True)
    q = queue.Queue()
    for i, j in enumerate(jobs):
        q.put((i, j))
    fails = []
    lock = threading.Lock()

    def worker(gpu):
        while True:
            try:
                i, cmd = q.get_nowait()
            except queue.Empty:
                return
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            lf = os.path.join(logdir, f"job{i:04d}.log")
            if dry:
                print(f"[gpu{gpu}] {cmd}")
                continue
            t0 = time.time()
            with open(lf, "w") as f:
                f.write(cmd + "\n\n")
                f.flush()
                rc = subprocess.call(cmd, shell=True, env=env, stdout=f,
                                     stderr=subprocess.STDOUT)
            with lock:
                tail = ""
                try:
                    tail = open(lf).read().strip().split("\n")[-1][:160]
                except Exception:
                    pass
                print(f"[gpu{gpu}] job{i:04d} rc={rc} {time.time()-t0:.0f}s  {tail}",
                      flush=True)
                if rc != 0:
                    fails.append((i, cmd, lf))

    ths = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in ths: t.start()
    for t in ths: t.join()
    if fails:
        print(f"\n{len(fails)} FAILED:")
        for i, c, lf in fails[:10]:
            print(" ", lf)
    return fails
