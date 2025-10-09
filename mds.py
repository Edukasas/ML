import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.datasets import load_digits
from sklearn.metrics import euclidean_distances
import time

# Load sample data
digits = load_digits()
X = digits.data[:500]  # Use subset for faster computation
y = digits.target[:500]

# Calculate distance matrix (optional - MDS can work with raw data or distances)
distances = euclidean_distances(X)

# =============================================================================
# 1. DEFAULT PARAMETERS - Just use the defaults!
# =============================================================================
print("=" * 60)
print("DEFAULT PARAMETERS")
print("=" * 60)

# Don't specify parameters - let sklearn use its defaults
mds_default = MDS(random_state=42)  # Only set random_state for reproducibility

start = time.time()
X_default = mds_default.fit_transform(X)
default_time = time.time() - start

print(f"Stress: {mds_default.stress_:.2f}")
print(f"Time: {default_time:.2f}s")
print(f"Iterations: {mds_default.n_iter_}")
print("Default values used:")
print(f"  n_components=2, max_iter=300, eps=0.001, n_init=4")

# =============================================================================
# 2. QUIET PARAMETERS - Fast, minimal quality
# =============================================================================
print("\n" + "=" * 60)
print("QUIET PARAMETERS")
print("=" * 60)

mds_quiet = MDS(
    max_iter=50,           # Low iterations
    n_init=1,              # Single initialization
    random_state=42,
    verbose=0              # Suppress any output
)

start = time.time()
X_quiet = mds_quiet.fit_transform(X)
quiet_time = time.time() - start

print(f"Stress: {mds_quiet.stress_:.2f}")
print(f"Time: {quiet_time:.2f}s")
print(f"Iterations: {mds_quiet.n_iter_}")

# =============================================================================
# 3. OPTIMAL PARAMETERS - High quality, slower
# =============================================================================
print("\n" + "=" * 60)
print("OPTIMAL PARAMETERS")
print("=" * 60)

mds_optimal = MDS(
    n_components=2,
    max_iter=1000,         # High iterations
    eps=1e-6,              # Very low tolerance (precise convergence)
    n_init=10,             # Many initializations to avoid local minima
    n_jobs=-1,             # Parallel processing
    random_state=42,
    dissimilarity='euclidean',
    verbose=1              # Show progress
)

start = time.time()
X_optimal = mds_optimal.fit_transform(X)
optimal_time = time.time() - start

print(f"Stress: {mds_optimal.stress_:.2f}")
print(f"Time: {optimal_time:.2f}s")
print(f"Iterations: {mds_optimal.n_iter_}")

# =============================================================================
# 4. VISUALIZE RESULTS
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

configs = [
    (X_default, mds_default, "Default", default_time),
    (X_quiet, mds_quiet, "Quiet", quiet_time),
    (X_optimal, mds_optimal, "Optimal", optimal_time)
]

for ax, (X_proj, mds_obj, title, exec_time) in zip(axes, configs):
    scatter = ax.scatter(X_proj[:, 0], X_proj[:, 1], 
                        c=y, cmap='tab10', alpha=0.6, s=20)
    ax.set_title(f'{title} MDS\nStress: {mds_obj.stress_:.2f} | Time: {exec_time:.2f}s')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.grid(True, alpha=0.3)

plt.colorbar(scatter, ax=axes[-1], label='Digit')
plt.tight_layout()
plt.savefig('mds_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 5. PARAMETER SWEEP EXAMPLE
# =============================================================================
print("\n" + "=" * 60)
print("PARAMETER SWEEP: Testing different n_init values")
print("=" * 60)

n_init_values = [1, 2, 5, 10, 20]
stresses = []
times = []

for n_init in n_init_values:
    mds = MDS(n_components=2, n_init=n_init, max_iter=300, 
              random_state=42, verbose=0)
    start = time.time()
    mds.fit_transform(X)
    times.append(time.time() - start)
    stresses.append(mds.stress_)
    print(f"n_init={n_init:2d} | Stress: {mds.stress_:8.2f} | Time: {times[-1]:.2f}s")

# Plot parameter sweep
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(n_init_values, stresses, 'o-', linewidth=2, markersize=8)
ax1.set_xlabel('n_init (number of initializations)')
ax1.set_ylabel('Stress')
ax1.set_title('Stress vs n_init')
ax1.grid(True, alpha=0.3)

ax2.plot(n_init_values, times, 's-', linewidth=2, markersize=8, color='orange')
ax2.set_xlabel('n_init (number of initializations)')
ax2.set_ylabel('Execution Time (s)')
ax2.set_title('Time vs n_init')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mds_parameter_sweep.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 6. ADVANCED: Using precomputed distance matrix
# =============================================================================
print("\n" + "=" * 60)
print("USING PRECOMPUTED DISTANCE MATRIX")
print("=" * 60)

mds_precomputed = MDS(
    n_components=2,
    dissimilarity='precomputed',  # Key parameter!
    max_iter=300,
    n_init=4,
    random_state=42,
    verbose=0
)

start = time.time()
X_precomputed = mds_precomputed.fit_transform(distances)
precomp_time = time.time() - start

print(f"Stress: {mds_precomputed.stress_:.2f}")
print(f"Time: {precomp_time:.2f}s")

print("\n" + "=" * 60)
print("KEY PARAMETERS TO EXPERIMENT WITH:")
print("=" * 60)
print("""
1. n_components: Number of dimensions (typically 2 or 3 for visualization)
2. n_init: Number of random initializations (higher = better, slower)
3. max_iter: Maximum iterations (higher = more convergence chances)
4. eps: Convergence tolerance (lower = more precise, slower)
5. dissimilarity: 'euclidean' or 'precomputed'
6. metric: True (metric MDS) or False (non-metric MDS)
7. n_jobs: Number of parallel jobs (-1 = use all CPUs)
8. random_state: For reproducibility
""")