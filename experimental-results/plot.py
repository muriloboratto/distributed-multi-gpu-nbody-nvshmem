#!/usr/bin/env python3

import re
import matplotlib.pyplot as plt

files = {
    "MPI": "result--32768-262144-1node-4GPUs-MMMMMMMM.txt",
    "CUDA-Aware MPI": "result--32768-262144-1node-4GPUs-CCCCCCCC.txt",
    "NCCL": "result--32768-262144-1node-4GPUs-NNNNNNNN.txt",
    "NVSHMEM": "result--32768-262144-1node-4GPUs-SSSSSSSS.txt",
}


def read_results(filename):
    results = {}

    pattern = re.compile(
        r"N-Body\s*\|\s*"
        r"Libraries=([MCNS]+)\s*\|\s*"
        r"Bodies=(\d+)\s*\|\s*"
        r"Average Time \(s\):\s*([0-9.eE+-]+)"
    )

    with open(filename, "r") as file:
        for line in file:
            match = pattern.search(line)

            if match:
                number_of_bodies = int(match.group(2))
                execution_time = float(match.group(3))

                results[number_of_bodies] = execution_time

    return results


data = {}

for library, filename in files.items():
    data[library] = read_results(filename)


common_sizes = set(data["MPI"].keys())

for library in data:
    common_sizes &= set(data[library].keys())

body_sizes = sorted(common_sizes)

if not body_sizes:
    raise RuntimeError(
        "No common N-body sizes were found in all result files."
    )


print("\nExecution Time (seconds)\n")

print(
    f"{'Bodies':>10}"
    f"{'MPI':>12}"
    f"{'CUDA-Aware':>15}"
    f"{'NCCL':>12}"
    f"{'NVSHMEM':>12}"
)

for size in body_sizes:
    print(
        f"{size:>10}"
        f"{data['MPI'][size]:>12.6f}"
        f"{data['CUDA-Aware MPI'][size]:>15.6f}"
        f"{data['NCCL'][size]:>12.6f}"
        f"{data['NVSHMEM'][size]:>12.6f}"
    )


plt.figure(figsize=(9, 6))

for library in files:
    times = [data[library][size] for size in body_sizes]

    plt.plot(
        body_sizes,
        times,
        marker="o",
        linewidth=2,
        label=library
    )

plt.xlabel("Number of Bodies")
plt.ylabel("Execution Time (s)")
plt.title(
    "N-Body — Execution Time\n"
    "1 Node / 4 GPUs"
)

plt.xticks(body_sizes)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()

plt.savefig(
    "nbody_execution_time.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


speedup = {}

for library in files:
    speedup[library] = []

    for size in body_sizes:
        mpi_time = data["MPI"][size]
        library_time = data[library][size]

        speedup_value = mpi_time / library_time

        speedup[library].append(speedup_value)


print("\nSpeedup relative to MPI\n")

print(
    f"{'Bodies':>10}"
    f"{'MPI':>12}"
    f"{'CUDA-Aware':>15}"
    f"{'NCCL':>12}"
    f"{'NVSHMEM':>12}"
)

for i, size in enumerate(body_sizes):
    print(
        f"{size:>10}"
        f"{speedup['MPI'][i]:>12.3f}"
        f"{speedup['CUDA-Aware MPI'][i]:>15.3f}"
        f"{speedup['NCCL'][i]:>12.3f}"
        f"{speedup['NVSHMEM'][i]:>12.3f}"
    )


plt.figure(figsize=(9, 6))

for library in files:
    plt.plot(
        body_sizes,
        speedup[library],
        marker="o",
        linewidth=2,
        label=library
    )

plt.axhline(
    y=1.0,
    linestyle="--",
    linewidth=1
)

plt.xlabel("Number of Bodies")
plt.ylabel("Speedup")
plt.title(
    "N-Body — Speedup Relative to MPI\n"
    "1 Node / 4 GPUs"
)

plt.xticks(body_sizes)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()

plt.savefig(
    "nbody_speedup.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nGenerated files:")
print("  nbody_execution_time.png")
print("  nbody_speedup.png")