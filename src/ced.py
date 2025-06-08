import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from pathlib import Path

recordings = [
    "wall1",
    "wall2",
    "wall3"
]
algorithms = ['BeamformerBase', 'BeamformerFunctional', 'BeamformerMusic', 'BeamformerCapon']

# Porównanie dla każdego algorytmu
for algo in algorithms:
    fig, ax = plt.subplots()
    ax.set_title(f"Cumulative error distribution ({algo})")

    for rec in recordings:
        dist_file = Path(f"../results/dist/{rec}_{algo}_dist_focuspoints.npy")
        dist = np.load(dist_file)
        hist, bins = np.histogram(dist, bins=100, range=(0, 100))
        ax.plot(bins[:-1], hist.cumsum() / hist.sum(), label=f"{rec}")

    ax.set_xlabel("Distance [px]")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.grid()
    ax.legend(loc='lower right')
    plt.savefig(Path("../results/plots") / f"ced_{algo}.svg", format='svg')
    plt.close()

# Porównanie wszystkich algorytmów dla jednego nagrania
# wall1
fig, ax = plt.subplots()
ax.set_title("Cumulative error distribution (all algorithms - wall1)")
for algo in algorithms:
    dist_file = Path(f"../results/dist/wall1_{algo}_dist_focuspoints.npy")
    try:
        dist = np.load(dist_file)
    except FileNotFoundError:
        print(f"Plik {dist_file} nie istnieje. Pomijam {algo}.")
        continue
    hist, bins = np.histogram(dist, bins=100, range=(0, 100))
    ax.plot(bins[:-1], hist.cumsum() / hist.sum(), label=algo)

ax.set_xlabel("Distance [px]")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.grid()
ax.legend(loc='lower right')
plt.savefig(Path("../results/plots") / "ced_all_wall1.svg", format='svg')
plt.close()

#wall2
fig, ax = plt.subplots()
ax.set_title("Cumulative error distribution (all algorithms - wall2)")
for algo in algorithms:
    dist_file = Path(f"../results/dist/wall2_{algo}_dist_focuspoints.npy")
    try:
        dist = np.load(dist_file)
    except FileNotFoundError:
        print(f"Plik {dist_file} nie istnieje. Pomijam {algo}.")
        continue
    hist, bins = np.histogram(dist, bins=100, range=(0, 100))
    ax.plot(bins[:-1], hist.cumsum() / hist.sum(), label=algo)

ax.set_xlabel("Distance [px]")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.grid()
ax.legend(loc='lower right')
plt.savefig(Path("../results/plots") / "ced_all_wall2.svg", format='svg')
plt.close()

#wall3
fig, ax = plt.subplots()
ax.set_title("Cumulative error distribution (all algorithms - wall3)")
for algo in algorithms:
    dist_file = Path(f"../results/dist/wall3_{algo}_dist_focuspoints.npy")
    try:
        dist = np.load(dist_file)
    except FileNotFoundError:
        print(f"Plik {dist_file} nie istnieje. Pomijam {algo}.")
        continue
    hist, bins = np.histogram(dist, bins=100, range=(0, 100))
    ax.plot(bins[:-1], hist.cumsum() / hist.sum(), label=algo)

ax.set_xlabel("Distance [px]")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.grid()
ax.legend(loc='lower right')
plt.savefig(Path("../results/plots") / "ced_all_wall3.svg", format='svg')
plt.close()