import subprocess
import time
import sys
import json
import random

def is_prime(n, k=5):
    if n < 2: return False
    for p in [2,3,5,7,11,13,17,19,23,29]:
        if n % p == 0: return n == p
    s, d = 0, n-1
    while d % 2 == 0:
        s, d = s+1, d//2
    for i in range(k):
        x = pow(random.randint(2, n-2), d, n)
        if x == 1 or x == n-1: continue
        for r in range(1, s):
            x = (x * x) % n
            if x == n-1: break
        else: return False
    return True

def generate_prime(bits):
    while True:
        p = random.getrandbits(bits) | (1 << (bits - 1)) | 1
        if is_prime(p): return p

def generate_semiprime(bits):
    p = generate_prime(bits // 2)
    q = generate_prime(bits - (bits // 2))
    return p * q

print("=== Compiling tools ===")
subprocess.run(["cargo", "build", "--release", "-p", "rust-nfs"], cwd=".", capture_output=True)

# Generate numbers
# smooth-pilatte currently works well up to ~40-64 bits.
# rust-nfs starts shining around 60-120 bits.
# cado-nfs minimum default parameter is for c30 (approx 100 bits).
sizes_to_test = [40, 60, 96]
numbers = {bits: generate_semiprime(bits) for bits in sizes_to_test}

print("\n=== Target Numbers ===")
for bits, n in numbers.items():
    print(f"{bits}-bit: {n}")

def run_rust_nfs(n):
    start = time.time()
    try:
        res = subprocess.run(["cargo", "run", "--release", "-p", "rust-nfs", "--", "--factor", str(n)],
                             capture_output=True, text=True, timeout=120)
        elapsed = time.time() - start
        success = "Factor:" in res.stderr or '"factor": "' in res.stdout
        return elapsed, "Success" if success else "Failed"
    except subprocess.TimeoutExpired:
        return 120, "Timeout"

def run_cado_nfs(n):
    # For small numbers, CADO doesn't have default params. We'll try to run it.
    start = time.time()
    try:
        # source venv and run
        cmd = f"cd cado-nfs && source cado-nfs.venv/bin/activate && ./cado-nfs.py {n}"
        res = subprocess.run(cmd, shell=True, executable='/bin/bash', capture_output=True, text=True, timeout=120)
        elapsed = time.time() - start
        if "no parameter file found" in res.stderr:
            return 0, "No Params"
        success = "Info: factor" in res.stdout or "factor" in res.stderr
        # CADO prints factors at the end of stdout
        return elapsed, "Success" if (str(n) not in res.stdout.split('\n')[-1] and res.returncode==0) else "Check"
    except subprocess.TimeoutExpired:
        return 120, "Timeout"

print("\n=== Benchmark Results ===")
print(f"{'Bits':<6} | {'rust-nfs':<15} | {'cado-nfs':<15}")
print("-" * 42)

for bits, n in numbers.items():
    r_nfs_time, r_nfs_status = run_rust_nfs(n)

    # CADO requires >= c30 (approx 96-100 bits) to run out of the box
    if bits >= 96:
        c_nfs_time, c_nfs_status = run_cado_nfs(n)
        cado_disp = f"{c_nfs_time:.2f}s ({c_nfs_status})"
    else:
        cado_disp = "N/A (too small)"

    rust_disp = f"{r_nfs_time:.2f}s ({r_nfs_status})"

    print(f"{bits:<6} | {rust_disp:<15} | {cado_disp:<15}")
