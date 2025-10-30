from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

# HYPERPARAMETERS - CENTRALIZED CONFIGURATION ------------------------------------------------------------------------------------------------

# Random seed for reproducibility
RANDOM_STATE = 42

# QUBO Matrix Parameters
W1 = 1.0              # Weight for feature relevance
W2 = 0.5              # Weight for redundancy penalty
LAMBDA_CORR = 1.0     # Correlation penalty multiplier
PENALTY_K = 10.0      # Penalty weight for violating k-constraint

# Simulated Annealing Parameters
T_INIT = 100          # Initial temperature
T_MIN = 0.01          # Minimum temperature
ALPHA = 0.95          # Cooling rate
MAX_ITER = 1000       # Iterations per temperature

# Search Configuration
K_MIN = 5             # Minimum k to test
K_MAX = 50            # Maximum k to test
K_STEP_COARSE = 5     # Step size for coarse search
FINE_SEARCH_RADIUS = 3  # Search radius around threshold k

# Optimal k Detection
THRESHOLD_PCT = 0.98      # Minimum percentage of baseline (98%)
MIN_IMPROVEMENT = 0.002   # Minimum improvement threshold (0.2%)

# Random Forest Parameters
RF_N_ESTIMATORS = 100     # Number of trees in Random Forest
RF_N_JOBS = -1            # Number of parallel jobs (-1 = all cores)

# Cross-Validation Parameters
CV_N_SPLITS = 5           # Number of CV folds
CV_SHUFFLE = True         # Shuffle data before splitting

# Visualization Parameters
FIG_DPI = 300            # DPI for saved figures

# END OF HYPERPARAMETERS -------------------------------------------------------------------

# LOAD DATA -------------------------------------------------------------------------------------------------------------------------

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
mi_scores = mutual_info_classif(X_scaled, y, random_state=RANDOM_STATE)
mi_normalized = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
print("Mutual information computation complete.\n")

# Correlation matrix (redundancy)
print("Computing correlation matrix...")
corr_matrix = np.corrcoef(X_scaled.T)
print("Correlation matrix computed.\n")

# BUILD QUBO MATRIX WITH CONSTRAINT FOR EXACTLY k FEATURES -------------------------------------------------------------------------------------------------------------------------

def build_qubo_with_k_constraint(n_features, mi_normalized, corr_matrix, k,
                                 w1=W1, w2=W2, lambda_corr=LAMBDA_CORR, penalty_k=PENALTY_K):
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

# SIMULATED ANNEALING WITH k-CONSTRAINT -------------------------------------------------------------------------------------------------------------------------

def compute_energy(x, Q):
    """Compute QUBO energy: E = x^T Q x"""
    return x.T @ Q @ x


def simulated_annealing_k_features(Q, k, T_init=T_INIT, T_min=T_MIN, alpha=ALPHA,
                                   max_iter=MAX_ITER, random_state=RANDOM_STATE):
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

# STEP 1: COMPUTE BASELINE (ALL FEATURES) -------------------------------------------------------------------------------------------------------------------------

print("=" * 70)
print("STEP 1: COMPUTING BASELINE (ALL 50 FEATURES)")
print("=" * 70 + "\n")

clf_baseline = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=RF_N_JOBS)
cv = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=CV_SHUFFLE, random_state=RANDOM_STATE)

start_time = time()
baseline_scores = cross_val_score(clf_baseline, X_scaled, y, cv=cv, 
                                  scoring='accuracy', n_jobs=RF_N_JOBS)
baseline_time = time() - start_time

baseline_acc = baseline_scores.mean()
baseline_std = baseline_scores.std()

print(f"Baseline Accuracy: {baseline_acc:.4f} ± {baseline_std:.4f}")
print(f"Training time: {baseline_time:.2f}s\n")

# STEP 2: AUTOMATIC k SELECTION FUNCTION -------------------------------------------------------------------------------------------------------------------------

def find_optimal_k(k_values, accuracies, baseline_acc, 
                   threshold_pct=THRESHOLD_PCT, min_improvement=MIN_IMPROVEMENT):
    """
    Two-stage approach to find optimal k:
    1. Find all k that achieve threshold_pct of baseline
    2. Among those, find where improvements drop below min_improvement
    
    Args:
        k_values: List of k values tested
        accuracies: List of accuracies for each k
        baseline_acc: Baseline accuracy (all features)
        threshold_pct: Minimum percentage of baseline to achieve (default: 0.98 = 98%)
        min_improvement: Minimum improvement required to continue (default: 0.002 = 0.2%)
    
    Returns:
        optimal_k: The optimal number of features
    """
    target = threshold_pct * baseline_acc
    
    print(f"\n{'='*70}")
    print(f"FINDING OPTIMAL k")
    print(f"{'='*70}")
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    print(f"Target ({threshold_pct*100:.0f}% of baseline): {target:.4f}")
    print(f"Minimum improvement threshold: {min_improvement:.4f} ({min_improvement*100:.2f}%)\n")
    
    # Stage 1: Filter by threshold -------------------------------------------------------------------------------------------------------------------------
    valid_k = [(k, acc) for k, acc in zip(k_values, accuracies) if acc >= target]
    
    if not valid_k:
        print(f"⚠️  WARNING: Never reached {threshold_pct*100:.0f}% of baseline!")
        print(f"   Target: {target:.4f}, Best achieved: {max(accuracies):.4f}")
        # Return k with best accuracy
        best_idx = np.argmax(accuracies)
        optimal_k = k_values[best_idx]
        print(f"   Returning k={optimal_k} with best accuracy: {accuracies[best_idx]:.4f}\n")
        return optimal_k
    
    min_k_meeting_threshold = min(valid_k, key=lambda x: x[0])[0]
    print(f"✓ Minimum k meeting {threshold_pct*100:.0f}% threshold: {min_k_meeting_threshold}")
    print(f"  Accuracy: {[acc for k, acc in valid_k if k == min_k_meeting_threshold][0]:.4f}\n")
    
    # Stage 2: Find where improvements plateau -------------------------------------------------------------------------------------------------------------------------
    print("Analyzing improvement rates:")
    improvements = []
    for i in range(len(accuracies) - 1):
        improvement = accuracies[i+1] - accuracies[i]
        improvements.append(improvement)
        status = "✓" if improvement >= min_improvement else "✗"
        print(f"  {status} k={k_values[i]:2d}→{k_values[i+1]:2d}: "
              f"Δ={improvement:+.4f} ({improvement*100:+.2f}%)")
    
    # Find first k where improvement drops below threshold
    for i, improvement in enumerate(improvements):
        k = k_values[i]
        if k >= min_k_meeting_threshold and improvement < min_improvement:
            print(f"\n✓ Diminishing returns detected at k={k}")
            print(f"  Improvement to next k: {improvement:.4f} < threshold {min_improvement:.4f}")
            return k
    
    print(f"\n✓ No clear diminishing point, using minimum k: {min_k_meeting_threshold}")
    return min_k_meeting_threshold

# HELPER FUNCTION TO EVALUATE A SINGLE k VALUE -------------------------------------------------------------------------------------------------------------------------

def evaluate_k(k, X_scaled, y, mi_normalized, corr_matrix, cv):
    """
    Evaluate QUBO feature selection for a specific k value.
    Returns a dictionary with results.
    """
    print(f"\n{'='*70}")
    print(f"Testing k = {k} features")
    print(f"{'='*70}\n")

    # Build QUBO matrix for this k
    Q = build_qubo_with_k_constraint(X_scaled.shape[1], mi_normalized, corr_matrix, k)

    # Run SA
    start_time = time()
    best_features, best_energy, sa_history = simulated_annealing_k_features(Q, k=k)
    sa_time = time() - start_time

    # Verify constraint
    assert best_features.sum() == k, f"Constraint violated! Got {best_features.sum()} features instead of {k}"

    # Get selected features
    selected_indices = np.where(best_features == 1)[0]
    X_selected = X_scaled[:, selected_indices]

    # Evaluate with cross-validation
    clf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=RF_N_JOBS)
    cv_scores = cross_val_score(clf, X_selected, y, cv=cv, scoring='accuracy', n_jobs=RF_N_JOBS)

    result = {
        'k': k,
        'energy': best_energy,
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'time': sa_time,
        'selected_indices': selected_indices,
        'selected_features': best_features.copy(),
        'history': sa_history
    }

    print(f"\nResults for k={k}:")
    print(f"  Energy: {best_energy:.4f}")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Time: {sa_time:.2f}s")
    print(f"  Selected features: {sorted(selected_indices.tolist())}")
    
    return result

# STEP 3: TWO-PHASE k SEARCH (COARSE + FINE) -------------------------------------------------------------------------------------------------------------------------

print("=" * 70)
print("STEP 2: TWO-PHASE SEARCH FOR OPTIMAL k")
print("=" * 70 + "\n")

print(f"QUBO Hyperparameters:")
print(f"  w1 (relevance weight): {W1}")
print(f"  w2 (redundancy weight): {W2}")
print(f"  lambda_corr: {LAMBDA_CORR}")
print(f"  penalty_k: {PENALTY_K}\n")

# PHASE 1: COARSE SEARCH (steps of 5) WITH EARLY STOPPING -------------------------------------------------------------------------------------------------------------------------

print("=" * 70)
print("PHASE 1: COARSE SEARCH (step size = 5)")
print("=" * 70 + "\n")

k_values_coarse = list(range(K_MIN, K_MAX + 1, K_STEP_COARSE))
threshold_target = THRESHOLD_PCT * baseline_acc

print(f"Planned k values: {k_values_coarse}")
print(f"Stopping criterion: Accuracy ≥ {THRESHOLD_PCT*100:.0f}% of baseline ({threshold_target:.4f})\n")

results = []
k_threshold_met = None

for k in k_values_coarse:
    result = evaluate_k(k, X_scaled, y, mi_normalized, corr_matrix, cv)
    results.append(result)
    
    # Check if we've reached threshold
    if result['cv_accuracy_mean'] >= threshold_target and k_threshold_met is None:
        k_threshold_met = k
        print(f"\n✓ THRESHOLD REACHED at k={k}")
        print(f"  Accuracy: {result['cv_accuracy_mean']:.4f} ≥ {threshold_target:.4f}")
        print(f"  Stopping coarse search, moving to fine-tuning...\n")
        break

if k_threshold_met is None:
    print(f"\n⚠️  WARNING: Never reached {THRESHOLD_PCT*100:.0f}% threshold in coarse search")
    print(f"  Best k found: k={results[-1]['k']} with {results[-1]['cv_accuracy_mean']:.4f}")
    k_threshold_met = results[-1]['k']

# Find approximate optimal k from coarse search
accuracies_coarse = [r['cv_accuracy_mean'] for r in results]
k_values_coarse_tested = [r['k'] for r in results]

print(f"\n{'='*70}")
print(f"PHASE 1 RESULT: Threshold met at k={k_threshold_met}")
print(f"{'='*70}\n")

# PHASE 2: FINE SEARCH (steps of 1) - Search 3 below and 3 above -------------------------------------------------------------------------------------------------------------------------

print("=" * 70)
print("PHASE 2: FINE SEARCH (step size = 1)")
print("=" * 70 + "\n")

# Define fine search range
k_min_fine = max(1, k_threshold_met - FINE_SEARCH_RADIUS)
k_max_fine = min(K_MAX, k_threshold_met + FINE_SEARCH_RADIUS)

# Only test k values we haven't tested yet
k_values_fine = [k for k in range(k_min_fine, k_max_fine + 1) 
                 if k not in k_values_coarse_tested]

if k_values_fine:
    print(f"Fine-tuning around k={k_threshold_met}")
    print(f"Search range: [{k_min_fine}, {k_max_fine}]")
    print(f"Testing additional k values: {k_values_fine}\n")
    
    for k in k_values_fine:
        result = evaluate_k(k, X_scaled, y, mi_normalized, corr_matrix, cv)
        results.append(result)
    
    # Sort results by k for easier analysis
    results.sort(key=lambda x: x['k'])
else:
    print(f"No fine-tuning needed - k={k_threshold_met} is at boundary or all nearby values tested.\n")

# Get all k values and accuracies (sorted)
k_values_all = sorted([r['k'] for r in results])
accuracies_all = [r['cv_accuracy_mean'] for r in results if r['k'] in k_values_all]

# STEP 3: FIND MINIMUM k WITH DIMINISHING RETURNS -------------------------------------------------------------------------------------------------------------------------
print(f"\n{'='*70}")
print("FINDING MINIMUM k (FIRST INSTANCE OF DIMINISHING RETURNS)")
print(f"{'='*70}\n")

# Get all k values and accuracies (sorted)
k_values_all = sorted([r['k'] for r in results])
accuracies_all_dict = {r['k']: r['cv_accuracy_mean'] for r in results}
accuracies_all = [accuracies_all_dict[k] for k in k_values_all]

print(f"Baseline accuracy: {baseline_acc:.4f}")
print(f"Target ({THRESHOLD_PCT*100:.0f}% of baseline): {threshold_target:.4f}")
print(f"Diminishing returns threshold: {MIN_IMPROVEMENT:.4f} ({MIN_IMPROVEMENT*100:.2f}%)\n")

print("Analyzing improvements:")
optimal_k = None

for i in range(len(k_values_all) - 1):
    k_current = k_values_all[i]
    k_next = k_values_all[i + 1]
    acc_current = accuracies_all[i]
    acc_next = accuracies_all[i + 1]
    improvement = acc_next - acc_current
    
    # Check if current k meets threshold
    meets_threshold = acc_current >= threshold_target
    status_threshold = "✓" if meets_threshold else "✗"
    
    # Check if improvement is below threshold
    low_improvement = improvement < MIN_IMPROVEMENT
    status_improvement = "✗" if low_improvement else "✓"
    
    print(f"  {status_threshold} k={k_current:2d}: {acc_current:.4f} "
          f"→ k={k_next:2d}: {acc_next:.4f} "
          f"(Δ={improvement:+.4f}) {status_improvement}")
    
    # Find first k that meets threshold AND has diminishing returns
    if meets_threshold and low_improvement and optimal_k is None:
        optimal_k = k_current
        print(f"\n✓ OPTIMAL k FOUND: {optimal_k}")
        print(f"  Accuracy: {acc_current:.4f} (≥ {threshold_target:.4f})")
        print(f"  Next improvement: {improvement:.4f} < {MIN_IMPROVEMENT:.4f}")
        print(f"  This is the minimum features needed!")
        break

# If no diminishing returns found, use the first k that meets threshold
if optimal_k is None:
    for i, k in enumerate(k_values_all):
        if accuracies_all[i] >= threshold_target:
            optimal_k = k
            print(f"\n✓ OPTIMAL k: {optimal_k} (first to meet threshold)")
            print(f"  Accuracy: {accuracies_all[i]:.4f}")
            print(f"  Note: No clear diminishing returns detected")
            break

# Fallback: if threshold never met, use best accuracy
if optimal_k is None:
    best_idx = np.argmax(accuracies_all)
    optimal_k = k_values_all[best_idx]
    print(f"\n⚠️  Threshold not met! Using k with best accuracy: {optimal_k}")
    print(f"  Accuracy: {accuracies_all[best_idx]:.4f}")

# Get results for optimal k
best_result = [r for r in results if r['k'] == optimal_k][0]

print(f"\n{'='*70}")
print("OPTIMAL CONFIGURATION")
print(f"{'='*70}")
print(f"\n✓ Optimal k: {optimal_k}")
print(f"  CV Accuracy: {best_result['cv_accuracy_mean']:.4f} (+/- {best_result['cv_accuracy_std']:.4f})")
print(f"  Baseline Accuracy: {baseline_acc:.4f} (+/- {baseline_std:.4f})")
print(f"  Performance retention: {(best_result['cv_accuracy_mean'] / baseline_acc) * 100:.2f}%")
print(f"  Feature reduction: {(1 - best_result['k'] / X.shape[1]) * 100:.1f}%")
print(f"  QUBO Energy: {best_result['energy']:.4f}")
print(f"  Time: {best_result['time']:.2f}s")
print(f"\n  Selected feature indices:")
print(f"  {sorted(best_result['selected_indices'].tolist())}")

# VISUALIZATIONS -------------------------------------------------------------------------------------------------------------------------

print(f"\n{'='*70}")
print("GENERATING VISUALIZATIONS")
print(f"{'='*70}\n")

fig = plt.figure(figsize=(18, 12))

# 1. Accuracy vs k with threshold lines
ax1 = plt.subplot(2, 3, 1)
k_vals = [r['k'] for r in results]
acc_vals = [r['cv_accuracy_mean'] for r in results]
acc_stds = [r['cv_accuracy_std'] for r in results]
ax1.errorbar(k_vals, acc_vals, yerr=acc_stds, marker='o', linewidth=2,
             markersize=8, capsize=5, color='#3b82f6')
ax1.axhline(baseline_acc, color='red', linestyle='--',
            linewidth=2, label='Baseline (all features)')
ax1.axhline(THRESHOLD_PCT*baseline_acc, color='orange', linestyle=':',
            linewidth=2, label=f'{THRESHOLD_PCT*100:.0f}% threshold')
ax1.axvline(optimal_k, color='green', linestyle='--',
            linewidth=2, alpha=0.7, label=f'Optimal k={optimal_k}')
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
ax2.axvline(optimal_k, color='green', linestyle='--',
            linewidth=2, alpha=0.7, label=f'Optimal k={optimal_k}')
ax2.set_xlabel('Number of Features (k)', fontweight='bold', fontsize=11)
ax2.set_ylabel('QUBO Energy', fontweight='bold', fontsize=11)
ax2.set_title('QUBO Energy vs k', fontweight='bold', fontsize=12)
ax2.legend()
ax2.grid(alpha=0.3)

# 3. SA Convergence for optimal k
ax3 = plt.subplot(2, 3, 3)
ax3.plot(best_result['history']['energy'], linewidth=2, color='#10b981')
ax3.set_xlabel('Temperature Step', fontweight='bold', fontsize=11)
ax3.set_ylabel('Best Energy', fontweight='bold', fontsize=11)
ax3.set_title(f'SA Convergence (k={optimal_k})', fontweight='bold', fontsize=12)
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
plt.savefig('selected_features_mi_scores.png', dpi=FIG_DPI, bbox_inches='tight')
print("✓ MI visualization saved: 'selected_features_mi_scores.png'")

plt.close('all')


# SAVE RESULTS -------------------------------------------------------------------------------------------------------------------------

print(f"\n{'='*70}")
print("SAVING RESULTS")
print(f"{'='*70}\n")

# Save summary -------------------------------------------------------------------------------------------------------------------------
with open('qubo_optimal_k_results.txt', 'w') as f:
    f.write("QUBO + SIMULATED ANNEALING: AUTOMATIC OPTIMAL-k FEATURE SELECTION\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Dataset: MiniBooNE\n")
    f.write(f"Total features: {X.shape[1]}\n")
    f.write(f"Total samples: {X.shape[0]}\n\n")

    f.write(f"QUBO Hyperparameters:\n")
    f.write(f"  w1 (relevance weight): {W1}\n")
    f.write(f"  w2 (redundancy weight): {W2}\n")
    f.write(f"  lambda_corr: {LAMBDA_CORR}\n")
    f.write(f"  penalty_k: {PENALTY_K}\n\n")
    
    f.write(f"Simulated Annealing Hyperparameters:\n")
    f.write(f"  T_init: {T_INIT}\n")
    f.write(f"  T_min: {T_MIN}\n")
    f.write(f"  alpha (cooling rate): {ALPHA}\n")
    f.write(f"  max_iter: {MAX_ITER}\n\n")
    
    f.write(f"Search Configuration:\n")
    f.write(f"  k_min: {K_MIN}\n")
    f.write(f"  k_max: {K_MAX}\n")
    f.write(f"  k_step_coarse: {K_STEP_COARSE}\n")
    f.write(f"  fine_search_radius: {FINE_SEARCH_RADIUS}\n")
    f.write(f"  threshold_pct: {THRESHOLD_PCT}\n")
    f.write(f"  min_improvement: {MIN_IMPROVEMENT}\n\n")
    
    f.write(f"Random Forest Hyperparameters:\n")
    f.write(f"  n_estimators: {RF_N_ESTIMATORS}\n")
    f.write(f"  random_state: {RANDOM_STATE}\n\n")
    
    f.write(f"Cross-Validation:\n")
    f.write(f"  n_splits: {CV_N_SPLITS}\n")
    f.write(f"  shuffle: {CV_SHUFFLE}\n\n")

    f.write(f"Tested k values: {k_vals}\n\n")
    
    f.write(f"BASELINE (ALL FEATURES):\n")
    f.write(f"  Features: 50\n")
    f.write(f"  CV Accuracy: {baseline_acc:.4f} (+/- {baseline_std:.4f})\n")
    f.write(f"  Time: {baseline_time:.2f}s\n\n")

    f.write(f"OPTIMAL CONFIGURATION:\n")
    f.write(f"  Optimal k: {optimal_k}\n")
    f.write(f"  CV Accuracy: {best_result['cv_accuracy_mean']:.4f} (+/- {best_result['cv_accuracy_std']:.4f})\n")
    f.write(f"  Performance retention: {(best_result['cv_accuracy_mean'] / baseline_acc) * 100:.2f}%\n")
    f.write(f"  Feature reduction: {(1 - best_result['k'] / X.shape[1]) * 100:.1f}%\n")
    f.write(f"  QUBO Energy: {best_result['energy']:.4f}\n")
    f.write(f"  Time: {best_result['time']:.2f}s\n\n")

    f.write(f"Selected feature indices:\n")
    f.write(f"  {sorted(best_result['selected_indices'].tolist())}\n\n")

    f.write(f"RESULTS FOR ALL k VALUES:\n")
    f.write("-" * 70 + "\n")
    for r in results:
        f.write(f"k={r['k']:2d}: Accuracy={r['cv_accuracy_mean']:.4f} (+/- {r['cv_accuracy_std']:.4f}), "
                f"Energy={r['energy']:.4f}, Time={r['time']:.2f}s\n")

print("✓ Results saved: 'qubo_optimal_k_results.txt'")

# Save selected features for optimal k -------------------------------------------------------------------------------------------------------------------------
np.savetxt(f'selected_features_k{optimal_k}.txt',
           best_result['selected_indices'], fmt='%d')
print(f"✓ Selected features saved: 'selected_features_k{optimal_k}.txt'")

print(f"\n{'='*70}")
print("COMPLETE!")
print(f"{'='*70}")
print(f"\nOptimal configuration: k={optimal_k} features")   
print(f"Accuracy: {best_result['cv_accuracy_mean']:.4f} (+/- {best_result['cv_accuracy_std']:.4f})")
print(f"Reduction: {(1 - optimal_k / X.shape[1]) * 100:.1f}%")
print(f"Retention: {(best_result['cv_accuracy_mean'] / baseline_acc) * 100:.2f}% of baseline")
plt.savefig('qubo_sa_optimal_k_results.png', dpi=FIG_DPI, bbox_inches='tight')
print("✓ Visualization saved: 'qubo_sa_optimal_k_results.png'")

# Additional plot: MI scores of selected features for optimal k -------------------------------------------------------------------------------------------------------------------------
fig2, ax = plt.subplots(figsize=(12, 6))
all_mi = mi_scores.copy()
selected_mask = np.zeros(len(all_mi), dtype=bool)
selected_mask[best_result['selected_indices']] = True

colors = ['#ef4444' if not selected_mask[i] else '#10b981' for i in range(len(all_mi))]
ax.bar(range(len(all_mi)), all_mi, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Feature Index', fontweight='bold', fontsize=12)
ax.set_ylabel('Mutual Information Score', fontweight='bold', fontsize=12)
ax.set_title(f'MI Scores: Selected (green) vs Not Selected (red) for k={optimal_k}',
             fontweight='bold', fontsize=13)
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()