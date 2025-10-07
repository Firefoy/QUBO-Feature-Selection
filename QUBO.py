from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
print("Loading MiniBooNE dataset...")
data = fetch_openml(name='MiniBooNE', version=1, parser='auto')
X = data.data.values
y = data.target.values
print(f"Data loaded: X shape = {X.shape}, y shape = {y.shape}\n")

# Standardize features
print("Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Feature standardization complete.\n")

# Mutual Information (relevance)
print("Computing mutual information scores...")
mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
mi_normalized = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
print("Mutual information computation complete.\n")

# Correlation matrix (redundancy)
print("Computing correlation matrix...")
corr_matrix = np.corrcoef(X_scaled.T)
print("Correlation matrix computed.\n")


# ============================================================================
# BUILD QUBO MATRIX WITH CONSTRAINT FOR EXACTLY k FEATURES
# ============================================================================
def build_qubo_with_k_constraint(n_features, mi_normalized, corr_matrix, k,
                                 w1=1.0, w2=0.5, lambda_corr=1.0, penalty_k=10.0):
    """
    Build QUBO matrix with constraint to select exactly k features.

    The constraint is: (sum(x_i) - k)^2 = 0
    Expanded: sum(x_i^2) - 2k*sum(x_i) + k^2
    Since x_i^2 = x_i (binary), this becomes: sum(x_i) - 2k*sum(x_i) + k^2

    Args:
        n_features: Number of features
        mi_normalized: Normalized mutual information scores
        corr_matrix: Feature correlation matrix
        k: Number of features to select
        w1: Weight for feature relevance
        w2: Weight for redundancy penalty
        lambda_corr: Correlation penalty multiplier
        penalty_k: Penalty weight for violating k-constraint

    Returns:
        Q: QUBO matrix
    """
    Q = np.zeros((n_features, n_features))

    # Original objective: maximize relevance, minimize redundancy
    # Diagonal terms: -relevance
    for i in range(n_features):
        Q[i, i] = -w1 * mi_normalized[i]

    # Off-diagonal terms: correlation penalty
    for i in range(n_features):
        for j in range(i + 1, n_features):
            penalty = w2 * lambda_corr * abs(corr_matrix[i, j])
            Q[i, j] = penalty
            Q[j, i] = penalty

    # Add constraint: (sum(x_i) - k)^2
    # Diagonal contribution: x_i * (1 - 2k)
    for i in range(n_features):
        Q[i, i] += penalty_k * (1 - 2 * k)

    # Off-diagonal contribution: 2 * x_i * x_j
    for i in range(n_features):
        for j in range(i + 1, n_features):
            Q[i, j] += penalty_k * 2
            Q[j, i] += penalty_k * 2

    return Q


# ============================================================================
# SIMULATED ANNEALING WITH k-CONSTRAINT
# ============================================================================
def compute_energy(x, Q):
    """Compute QUBO energy: E = x^T Q x"""
    return x.T @ Q @ x


def simulated_annealing_k_features(Q, k, T_init=100, T_min=0.01, alpha=0.95,
                                   max_iter=1000, random_state=42):
    """
    Simulated Annealing that enforces exactly k features selected.

    Args:
        Q: QUBO matrix
        k: Number of features to select
        T_init: Initial temperature
        T_min: Minimum temperature
        alpha: Cooling rate
        max_iter: Iterations per temperature
        random_state: Random seed

    Returns:
        best_solution: Binary vector with exactly k features selected
        best_energy: Minimum energy found
        history: Optimization history
    """
    np.random.seed(random_state)
    n = Q.shape[0]

    # Initialize with exactly k random features
    x_current = np.zeros(n, dtype=int)
    selected_indices = np.random.choice(n, k, replace=False)
    x_current[selected_indices] = 1
    energy_current = compute_energy(x_current, Q)

    # Track best solution
    x_best = x_current.copy()
    energy_best = energy_current

    # Track history
    history = {
        'energy': [],
        'temperature': [],
        'acceptance_rate': []
    }

    T = T_init
    iteration = 0

    print(f"Starting SA with k={k} features constraint")
    print(f"Initial energy: {energy_current:.4f}\n")

    while T > T_min:
        accepted = 0

        for _ in range(max_iter):
            # Generate neighbor by swapping one selected and one unselected feature
            x_new = x_current.copy()

            # Get indices of selected and unselected features
            selected = np.where(x_current == 1)[0]
            unselected = np.where(x_current == 0)[0]

            # Randomly pick one from each group and swap
            remove_idx = np.random.choice(selected)
            add_idx = np.random.choice(unselected)

            x_new[remove_idx] = 0
            x_new[add_idx] = 1

            # Verify constraint (should always be k)
            assert x_new.sum() == k, f"Constraint violated: {x_new.sum()} != {k}"

            # Compute energy of new solution
            energy_new = compute_energy(x_new, Q)

            # Energy difference
            delta_E = energy_new - energy_current

            # Metropolis acceptance criterion
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                x_current = x_new
                energy_current = energy_new
                accepted += 1
                print(energy_current)
                # Update best solution
                if energy_current < energy_best:
                    x_best = x_current.copy()
                    energy_best = energy_current

            iteration += 1

        # Record metrics
        acceptance_rate = accepted / max_iter
        history['energy'].append(energy_best)
        history['temperature'].append(T)
        history['acceptance_rate'].append(acceptance_rate)

        print(f"T={T:.4f}, Best Energy={energy_best:.4f}, "
              f"Features={x_best.sum()}, Accept Rate={acceptance_rate:.2%}")

        # Cool down
        T *= alpha

    return x_best, energy_best, history


# ============================================================================
# EVALUATE PERFORMANCE FOR DIFFERENT VALUES OF k
# ============================================================================
print("=" * 70)
print("EVALUATING DIFFERENT VALUES OF k")
print("=" * 70 + "\n")

# Hyperparameters for QUBO
w1 = 1.0
w2 = 0.5
lambda_corr = 1.0
penalty_k = 10.0

# Test different k values
k_values = [20,25,30]
results = []

for k in k_values:
    print(f"\n{'=' * 70}")
    print(f"Testing k = {k} features")
    print(f"{'=' * 70}\n")

    # Build QUBO matrix for this k
    Q = build_qubo_with_k_constraint(
        X.shape[1], mi_normalized, corr_matrix, k,
        w1=w1, w2=w2, lambda_corr=lambda_corr, penalty_k=penalty_k
    )

    # Run SA
    best_features, best_energy, sa_history = simulated_annealing_k_features(
        Q, k=k, T_init=100, T_min=0.01, alpha=0.95,
        max_iter=1000, random_state=42
    )

    # Verify constraint
    assert best_features.sum() == k, f"Constraint violated! Got {best_features.sum()} features instead of {k}"

    # Get selected features
    selected_indices = np.where(best_features == 1)[0]
    X_selected = X_scaled[:, selected_indices]

    # Evaluate with cross-validation
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_selected, y, cv=cv, scoring='accuracy', n_jobs=-1)

    result = {
        'k': k,
        'energy': best_energy,
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'selected_indices': selected_indices,
        'selected_features': best_features.copy(),
        'history': sa_history
    }
    results.append(result)

    print(f"\nResults for k={k}:")
    print(f"  Energy: {best_energy:.4f}")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Selected features: {sorted(selected_indices.tolist())}")

# ============================================================================
# FIND BEST k
# ============================================================================
print(f"\n{'=' * 70}")
print("SUMMARY: BEST k VALUE")
print(f"{'=' * 70}\n")

# Find best k by accuracy
best_idx = np.argmax([r['cv_accuracy_mean'] for r in results])
best_result = results[best_idx]

print(f"Best k: {best_result['k']}")
print(f"Best CV Accuracy: {best_result['cv_accuracy_mean']:.4f} (+/- {best_result['cv_accuracy_std']:.4f})")
print(f"QUBO Energy: {best_result['energy']:.4f}")
print(f"Selected features: {sorted(best_result['selected_indices'].tolist())}")

# Compare with baseline (all features)
print(f"\nBaseline (all {X.shape[1]} features):")
clf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
cv_baseline = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline_scores = cross_val_score(clf_baseline, X_scaled, y, cv=cv_baseline, scoring='accuracy', n_jobs=-1)
print(f"  CV Accuracy: {baseline_scores.mean():.4f} (+/- {baseline_scores.std():.4f})")

print(f"\nFeature reduction: {(1 - best_result['k'] / X.shape[1]) * 100:.1f}%")
print(f"Performance retention: {(best_result['cv_accuracy_mean'] / baseline_scores.mean()) * 100:.2f}%")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print(f"\n{'=' * 70}")
print("GENERATING VISUALIZATIONS")
print(f"{'=' * 70}\n")

fig = plt.figure(figsize=(18, 12))

# 1. Accuracy vs k
ax1 = plt.subplot(2, 3, 1)
k_vals = [r['k'] for r in results]
acc_vals = [r['cv_accuracy_mean'] for r in results]
acc_stds = [r['cv_accuracy_std'] for r in results]
ax1.errorbar(k_vals, acc_vals, yerr=acc_stds, marker='o', linewidth=2,
             markersize=8, capsize=5, color='#3b82f6')
ax1.axhline(baseline_scores.mean(), color='red', linestyle='--',
            linewidth=2, label='Baseline (all features)')
ax1.axvline(best_result['k'], color='green', linestyle='--',
            linewidth=2, alpha=0.7, label=f'Best k={best_result["k"]}')
ax1.set_xlabel('Number of Features (k)', fontweight='bold', fontsize=11)
ax1.set_ylabel('CV Accuracy', fontweight='bold', fontsize=11)
ax1.set_title('Accuracy vs Number of Features', fontweight='bold', fontsize=12)
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Energy vs k
ax2 = plt.subplot(2, 3, 2)
energy_vals = [r['energy'] for r in results]
ax2.plot(k_vals, energy_vals, marker='o', linewidth=2,
         markersize=8, color='#ef4444')
ax2.axvline(best_result['k'], color='green', linestyle='--',
            linewidth=2, alpha=0.7, label=f'Best k={best_result["k"]}')
ax2.set_xlabel('Number of Features (k)', fontweight='bold', fontsize=11)
ax2.set_ylabel('QUBO Energy', fontweight='bold', fontsize=11)
ax2.set_title('QUBO Energy vs k', fontweight='bold', fontsize=12)
ax2.legend()
ax2.grid(alpha=0.3)

# 3. SA Convergence for best k
ax3 = plt.subplot(2, 3, 3)
ax3.plot(best_result['history']['energy'], linewidth=2, color='#10b981')
ax3.set_xlabel('Temperature Step', fontweight='bold', fontsize=11)
ax3.set_ylabel('Best Energy', fontweight='bold', fontsize=11)
ax3.set_title(f'SA Convergence (k={best_result["k"]})', fontweight='bold', fontsize=12)
ax3.grid(alpha=0.3)

# 4. Temperature schedule
ax4 = plt.subplot(2, 3, 4)
ax4.plot(best_result['history']['temperature'], linewidth=2, color='#f59e0b')
ax4.set_xlabel('Temperature Step', fontweight='bold', fontsize=11)
ax4.set_ylabel('Temperature', fontweight='bold', fontsize=11)
ax4.set_title('Temperature Schedule', fontweight='bold', fontsize=12)
ax4.set_yscale('log')
ax4.grid(alpha=0.3)

# 5. Acceptance rate
ax5 = plt.subplot(2, 3, 5)
ax5.plot(best_result['history']['acceptance_rate'], linewidth=2,
         marker='o', markersize=4, color='#8b5cf6')
ax5.set_xlabel('Temperature Step', fontweight='bold', fontsize=11)
ax5.set_ylabel('Acceptance Rate', fontweight='bold', fontsize=11)
ax5.set_title('Acceptance Rate', fontweight='bold', fontsize=12)
ax5.grid(alpha=0.3)

# 6. Feature selection frequency across different k
ax6 = plt.subplot(2, 3, 6)
selection_matrix = np.array([r['selected_features'] for r in results])
feature_freq = selection_matrix.mean(axis=0)
top_30_idx = np.argsort(feature_freq)[-30:]
ax6.barh(range(30), feature_freq[top_30_idx], color='#14b8a6', alpha=0.7)
ax6.set_yticks(range(30))
ax6.set_yticklabels([f'F{i}' for i in top_30_idx], fontsize=8)
ax6.set_xlabel('Selection Frequency', fontweight='bold', fontsize=11)
ax6.set_title('Top 30 Most Frequently Selected Features', fontweight='bold', fontsize=12)
ax6.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('qubo_sa_top_k_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: 'qubo_sa_top_k_results.png'")

# Additional plot: MI scores of selected features for best k
fig2, ax = plt.subplots(figsize=(12, 6))
all_mi = mi_scores.copy()
selected_mask = np.zeros(len(all_mi), dtype=bool)
selected_mask[best_result['selected_indices']] = True

colors = ['#ef4444' if not selected_mask[i] else '#10b981' for i in range(len(all_mi))]
ax.bar(range(len(all_mi)), all_mi, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Feature Index', fontweight='bold', fontsize=12)
ax.set_ylabel('Mutual Information Score', fontweight='bold', fontsize=12)
ax.set_title(f'MI Scores: Selected (green) vs Not Selected (red) for k={best_result["k"]}',
             fontweight='bold', fontsize=13)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('selected_features_mi_scores.png', dpi=300, bbox_inches='tight')
print("✓ MI visualization saved: 'selected_features_mi_scores.png'")

plt.close('all')

# ============================================================================
# SAVE RESULTS
# ============================================================================
print(f"\n{'=' * 70}")
print("SAVING RESULTS")
print(f"{'=' * 70}\n")

# Save summary
with open('qubo_topk_results.txt', 'w') as f:
    f.write("QUBO + SIMULATED ANNEALING: TOP-k FEATURE SELECTION\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Dataset: MiniBooNE\n")
    f.write(f"Total features: {X.shape[1]}\n")
    f.write(f"Total samples: {X.shape[0]}\n\n")

    f.write(f"QUBO Hyperparameters:\n")
    f.write(f"  w1 (relevance weight): {w1}\n")
    f.write(f"  w2 (redundancy weight): {w2}\n")
    f.write(f"  lambda_corr: {lambda_corr}\n")
    f.write(f"  penalty_k: {penalty_k}\n\n")

    f.write(f"Tested k values: {k_values}\n\n")

    f.write(f"BEST CONFIGURATION:\n")
    f.write(f"  Best k: {best_result['k']}\n")
    f.write(f"  CV Accuracy: {best_result['cv_accuracy_mean']:.4f} (+/- {best_result['cv_accuracy_std']:.4f})\n")
    f.write(f"  Baseline Accuracy: {baseline_scores.mean():.4f} (+/- {baseline_scores.std():.4f})\n")
    f.write(f"  Performance retention: {(best_result['cv_accuracy_mean'] / baseline_scores.mean()) * 100:.2f}%\n")
    f.write(f"  Feature reduction: {(1 - best_result['k'] / X.shape[1]) * 100:.1f}%\n")
    f.write(f"  QUBO Energy: {best_result['energy']:.4f}\n\n")

    f.write(f"Selected feature indices:\n")
    f.write(f"  {sorted(best_result['selected_indices'].tolist())}\n\n")

    f.write(f"RESULTS FOR ALL k VALUES:\n")
    f.write("-" * 70 + "\n")
    for r in results:
        f.write(
            f"k={r['k']:2d}: Accuracy={r['cv_accuracy_mean']:.4f} (+/- {r['cv_accuracy_std']:.4f}), Energy={r['energy']:.4f}\n")

print("✓ Results saved: 'qubo_topk_results.txt'")

# Save selected features for best k
np.savetxt(f'selected_features_k{best_result["k"]}.txt',
           best_result['selected_indices'], fmt='%d')
print(f"✓ Selected features saved: 'selected_features_k{best_result['k']}.txt'")

print(f"\n{'=' * 70}")
print("COMPLETE!")
print(f"{'=' * 70}")
print(f"\nBest configuration: k={best_result['k']} features")
print(f"Accuracy: {best_result['cv_accuracy_mean']:.4f} (+/- {best_result['cv_accuracy_std']:.4f})")
print(f"Reduction: {(1 - best_result['k'] / X.shape[1]) * 100:.1f}%")