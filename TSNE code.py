from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = fetch_openml(name='MiniBooNE', version=1, parser='auto')

X = data.data
y = data.target.values

# Label features as feature_1, feature_2 etc
X.columns = [f"feature_{i+1}" for i in range(X.shape[1])]


le = LabelEncoder()
y_encoded = le.fit_transform(y)  # e.g., signal = 1, background = 0


subset_idx = np.random.choice(len(X), 5000, replace=False)
X_sub = X.iloc[subset_idx]
y_sub = y_encoded[subset_idx]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sub)

#This is the running of tsne
tsne = TSNE(n_components=2, perplexity=40, random_state=42, init="pca", n_iter=1000)
X_embedded = tsne.fit_transform(X_scaled)


plt.figure(figsize=(7,5))
plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y_sub, cmap='coolwarm', s=5, alpha=0.7)
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")
plt.title("t-SNE visualization of MiniBooNE dataset")
plt.colorbar(label="Particle label (0=background, 1=signal)")
plt.tight_layout()
plt.show()


#Feature correlation analysis
corrs = []
for i, fname in enumerate(X.columns):
    r1, _ = pearsonr(X_scaled[:, i], X_embedded[:, 0])  # correlation with dim 1
    r2, _ = pearsonr(X_scaled[:, i], X_embedded[:, 1])  # correlation with dim 2
    total_corr = abs(r1) + abs(r2)
    corrs.append((fname, float(r1), float(r2), total_corr))

corr_df = pd.DataFrame(corrs, columns=["feature", "corr_dim1", "corr_dim2", "sum_abs_corr"])
corr_df = corr_df.sort_values("sum_abs_corr", ascending=False).reset_index(drop=True)

#Top features 
print("\nTop 10 features influencing t-SNE layout:")
print(corr_df.head(10))
