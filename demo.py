import numpy as np
import matplotlib.pyplot as plt
import prad

FIELD = "data/instabilities/zpinch.bfld"

# ── 1. Run the simulation ─────────────────────────────────────────────────────

result = prad.run(
    FIELD,
    source="parallel",
    energy_MeV=14.7,
    n_particles=200_000,
    source_distance_mm=80.0,
    beam_radius_mm=40.0,
    detector_distance_mm=100.0,
    detector_size_mm=(500.0, 500.0),
    dt_ps=0.2,
    max_steps=25_000,
)

print(f"particles: {result.diagnostics['n_particles']:,}")
print(f"hits:      {result.diagnostics['n_hits']:,}  ({result.diagnostics['hit_fraction']:.1%})")
print(f"runtime:   {result.metadata['performance']['total_runtime_s']:.2f}s")
print(f"counts:    {result.raw_counts.shape}  dtype={result.raw_counts.dtype}")

# ── 2. Run all 5 instability fields and compare ───────────────────────────────

fields = {
    "Z-pinch":       "data/instabilities/zpinch.bfld",
    "Kink (weak)":   "data/instabilities/kink_weak.bfld",
    "Kink (strong)": "data/instabilities/kink_strong.bfld",
    "Sausage (weak)": "data/instabilities/sausage_weak.bfld",
    "Sausage (strong)": "data/instabilities/sausage_strong.bfld",
}

results = {}
for name, path in fields.items():
    print(f"running {name}...", flush=True)
    results[name] = prad.run(
        path,
        n_particles=50_000,
        dt_ps=0.2,
        max_steps=25_000,
    )

# ── 3. Plot ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, len(results), figsize=(18, 4))
fig.suptitle("Proton radiography — plasma instabilities", fontsize=13, y=1.02)

for ax, (name, res) in zip(axes, results.items()):
    counts = res.raw_counts.astype(np.float32)
    ax.imshow(np.log1p(counts), cmap="inferno", origin="lower")
    ax.set_title(name, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("demo_radiographs.png", dpi=150, bbox_inches="tight")
plt.show()
print("saved demo_radiographs.png")

# ── 4. Build a field from numpy and run ───────────────────────────────────────

# Simple uniform Bz field — protons bend in a circle
nx, ny, nz = 64, 64, 64
B = np.zeros((nx, ny, nz, 3), dtype=np.float32)
B[:, :, :, 2] = 5.0   # 5 Tesla uniform Bz

field = prad.Field.from_array(B, bounds_m=(-0.05, 0.05, -0.05, 0.05, -0.05, 0.05))

result_uniform = prad.run(
    field,
    n_particles=50_000,
    energy_MeV=14.7,
    dt_ps=0.2,
    max_steps=25_000,
)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Uniform 5T Bz field — circular deflection", fontsize=12)
for ax, (label, res) in zip(axes, [("zpinch", result), ("uniform Bz", result_uniform)]):
    counts = res.raw_counts.astype(np.float32)
    ax.imshow(np.log1p(counts), cmap="inferno", origin="lower")
    ax.set_title(label)
    ax.axis("off")
plt.tight_layout()
plt.savefig("demo_uniform_bz.png", dpi=150, bbox_inches="tight")
plt.show()
print("saved demo_uniform_bz.png")
