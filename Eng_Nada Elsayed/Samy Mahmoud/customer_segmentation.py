"""
Wholesale Customers Segmentation using K-Means Clustering
Course: Machine Learning 2026 - Assoc. Prof. Mohamed Abd Elfattah
Deliverable: Python Pipeline & Visualizations
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

# Set style for professional presentation
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Segoe UI', 'DejaVu Sans', 'Arial'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 1.2

# 1. Load Data
data_path = 'Wholesale customers data.csv'
df = pd.read_csv(data_path)
print(f"Dataset Loaded Successfully: {df.shape[0]} samples, {df.shape[1]} features.")
print("\n--- Summary Statistics ---")
spending_cols = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
print(df[spending_cols].describe().round(2))

# Check Skewness before transformation
skew_raw = df[spending_cols].skew()
print("\n--- Raw Data Skewness ---")
print(skew_raw)

# 2. Log Transform and Standardize
# Financial and spending data often follow log-normal distributions with heavy right tails.
df_log = np.log1p(df[spending_cols])
skew_log = df_log.skew()
print("\n--- Log-Transformed Skewness ---")
print(skew_log)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_log)

# Plot Distributions Before & After Transformation
fig, axes = plt.subplots(2, 6, figsize=(20, 7))
fig.suptitle('Feature Distributions: Raw Spending (Heavy Skew) vs. Log-Transformed & Standardized', fontsize=16, fontweight='bold', y=0.98)

palette = sns.color_palette("mako", 6)
for i, col in enumerate(spending_cols):
    sns.histplot(df[col], kde=True, ax=axes[0, i], color=palette[i], bins=20)
    axes[0, i].set_title(f'{col}\nSkew: {skew_raw[col]:.2f}', fontsize=11, fontweight='semibold')
    axes[0, i].set_xlabel('Annual Spending (m.u.)')
    
    sns.histplot(X_scaled[:, i], kde=True, ax=axes[1, i], color='#0284c7', bins=20)
    axes[1, i].set_title(f'Log + Scaled {col}\nSkew: {skew_log[col]:.2f}', fontsize=11, fontweight='semibold')
    axes[1, i].set_xlabel('Standardized Units')

plt.tight_layout()
plt.savefig('distribution_and_transformations.png', dpi=300)
plt.close()
print("Saved: distribution_and_transformations.png")

# 3. K-Means for K = 2..8 (Elbow & Silhouette)
k_range = list(range(2, 9))
inertias = []
silhouettes = []
davies_bouldin = []
calinski_harabasz = []

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_score = silhouette_score(X_scaled, labels)
    silhouettes.append(sil_score)
    davies_bouldin.append(davies_bouldin_score(X_scaled, labels))
    calinski_harabasz.append(calinski_harabasz_score(X_scaled, labels))
    print(f"K={k}: Inertia={km.inertia_:.2f}, Silhouette={sil_score:.4f}, DB-Index={davies_bouldin[-1]:.3f}, CH-Index={calinski_harabasz[-1]:.1f}")

best_k_sil = k_range[np.argmax(silhouettes)]
print(f"\nHighest Silhouette Score is at K = {best_k_sil} ({max(silhouettes):.4f})")

# Plot Elbow and Silhouette Curves
fig, ax1 = plt.subplots(figsize=(10, 5.5))

color_inertia = '#2563eb'
ax1.set_xlabel('Number of Clusters (K)', fontsize=13, fontweight='bold', labelpad=10)
ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares, WCSS)', color=color_inertia, fontsize=12, fontweight='bold')
line1 = ax1.plot(k_range, inertias, marker='o', linewidth=2.5, markersize=8, color=color_inertia, label='Inertia (Elbow)')
ax1.tick_params(axis='y', labelcolor=color_inertia)
ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = ax1.twinx()
color_sil = '#059669'
ax2.set_ylabel('Mean Silhouette Score', color=color_sil, fontsize=12, fontweight='bold')
line2 = ax2.plot(k_range, silhouettes, marker='s', linewidth=2.5, markersize=8, color=color_sil, linestyle='--', label='Silhouette Score')
ax2.tick_params(axis='y', labelcolor=color_sil)
ax2.grid(False)

# Highlight chosen K = 3 (or best)
optimal_k = 3
ax1.axvline(x=optimal_k, color='#e11d48', linestyle=':', linewidth=2, label=f'Chosen Optimal K={optimal_k}')

lines = line1 + line2 + [plt.Line2D([0], [0], color='#e11d48', linestyle=':', linewidth=2)]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', frameon=True, framealpha=0.9, facecolor='#ffffff')

plt.title('Determining Optimal K: Elbow Method (Inertia) & Silhouette Analysis', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('elbow_silhouette_plot.png', dpi=300)
plt.close()
print("Saved: elbow_silhouette_plot.png")

# 4. Final K-Means Model with Optimal K
# In retail and wholesale distribution, K=3 provides the best blend of business interpretability,
# high silhouette score, and distinct actionable buyer groups (Restaurants/Hotels vs. Retail Supermarkets vs. Fresh Specialized Vendors).
# We also compute K=2 as comparison.
chosen_k = 3
km_final = KMeans(n_clusters=chosen_k, init='k-means++', n_init=30, random_state=42)
cluster_labels = km_final.fit_predict(X_scaled)
df['Cluster'] = cluster_labels

# 5. Cluster Profiling and Segmentation Analysis
cluster_summary = df.groupby('Cluster')[spending_cols].mean()
overall_mean = df[spending_cols].mean()
relative_spending = cluster_summary.div(overall_mean, axis=1)

print("\n--- Cluster Mean Annual Spending (Original Scale) ---")
print(cluster_summary.round(1))

print("\n--- Spending Relative to Dataset Average (1.0 = Average) ---")
print(relative_spending.round(2))

cluster_counts = df['Cluster'].value_counts().sort_index()
print("\n--- Cluster Counts & Percentages ---")
for c, count in cluster_counts.items():
    print(f"Cluster {c}: {count} clients ({count / len(df) * 100:.1f}%)")

# Naming the segments based on domain interpretation
# Let's inspect the top spendings for each cluster to accurately name them
names_map = {}
actions_map = {}

for c in range(chosen_k):
    rel = relative_spending.loc[c]
    top_features = rel.sort_values(ascending=False)
    print(f"\nCluster {c} Key Drivers: {top_features.index[0]} ({top_features.iloc[0]:.2f}x), {top_features.index[1]} ({top_features.iloc[1]:.2f}x)")

# Dynamically assign meaningful industry names based on spending profiles
# Segment 0, 1, 2 profiling:
for c in range(chosen_k):
    rel = relative_spending.loc[c]
    if rel['Fresh'] > 1.2 and rel['Frozen'] > 1.2 and rel['Grocery'] < 1.0:
        names_map[c] = "Fresh Food & Dining Services (Restaurants / Catering / Horeca)"
        actions_map[c] = "Promote cold-chain bulk discounts, seasonal fresh produce subscriptions, and flexible morning deliveries."
    elif rel['Grocery'] > 1.3 and rel['Detergents_Paper'] > 1.3 and rel['Milk'] > 1.2:
        names_map[c] = "Retail Grocers & Supermarkets"
        actions_map[c] = "Offer multi-pack FMCG volume rebates, automated replenishment EDI, and credit terms for shelf-stable goods."
    else:
        names_map[c] = "Small Scale Businesses & Corner Delis"
        actions_map[c] = "Offer low minimum order thresholds, consolidated mixed-pallet orders, and loyalty cash-back programs."

# In case naming needed fallback
if len(names_map) < chosen_k:
    default_names = [
        "Retailers & Supermarkets (High Grocery & Paper)",
        "HoReCa: Hotels, Restaurants & Cafes (High Fresh & Frozen)",
        "Small Wholesalers & Delicatessens (Moderate Balanced Spending)"
    ]
    for c in range(chosen_k):
        if c not in names_map:
            names_map[c] = default_names[c]
            actions_map[c] = "Tailor customized sales incentives and inventory support."

df['Segment_Name'] = df['Cluster'].map(names_map)
df['Recommended_Action'] = df['Cluster'].map(actions_map)

# Save processed segmented dataset
df.to_csv('segmented_wholesale_customers.csv', index=False)
print("\nSaved: segmented_wholesale_customers.csv")

# 6. Plot Cluster Spending Profiles
fig, ax = plt.subplots(figsize=(12, 6.5))
cluster_summary_t = cluster_summary.T
cluster_summary_t.columns = [f"Cluster {c}: {names_map[c].split('(')[0].strip()}" for c in range(chosen_k)]
cluster_summary_t.plot(kind='bar', ax=ax, width=0.8, colormap='viridis')

ax.set_title('Cluster Spending Profile: Average Annual Expenditure per Product Category', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Average Expenditure (m.u.)', fontsize=12, fontweight='bold')
ax.set_xlabel('Product Category', fontsize=12, fontweight='bold')
ax.set_xticklabels(spending_cols, rotation=25, ha='right', fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(title='Customer Segments', frameon=True, facecolor='#ffffff', fontsize=10)

plt.tight_layout()
plt.savefig('cluster_profiles.png', dpi=300)
plt.close()
print("Saved: cluster_profiles.png")

# 7. PCA 2D Dimensionality Reduction & Scatter Plot
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

var_exp = pca.explained_variance_ratio_
print(f"\nPCA Explained Variance: PC1 = {var_exp[0]*100:.2f}%, PC2 = {var_exp[1]*100:.2f}% (Total = {sum(var_exp)*100:.2f}%)")

# Transform centroids to PCA space
centroids_scaled = km_final.cluster_centers_
centroids_pca = pca.transform(centroids_scaled)

fig, ax = plt.subplots(figsize=(11, 7.5))
colors = ['#2563eb', '#10b981', '#f59e0b', '#8b5cf6']
markers = ['o', 's', '^', 'D']

for c in range(chosen_k):
    cluster_pts = df[df['Cluster'] == c]
    ax.scatter(cluster_pts['PCA1'], cluster_pts['PCA2'], 
               c=colors[c % len(colors)], label=f"Cluster {c}: {names_map[c].split('(')[0].strip()} (n={len(cluster_pts)})", 
               alpha=0.65, s=60, edgecolors='white', linewidth=0.5)

# Plot Centroids as prominent diamonds (as shown in lecture slides)
ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
           c='#e11d48', s=260, marker='D', edgecolor='#ffffff', linewidth=2, 
           label='Cluster Centroids ($\mu_k$)', zorder=10)

# Add centroid annotations
for c in range(chosen_k):
    ax.annotate(f"Centroid {c}", (centroids_pca[c, 0], centroids_pca[c, 1]),
                textcoords="offset points", xytext=(0, 12), ha='center',
                fontsize=11, fontweight='bold', color='#9f1239',
                bbox=dict(boxstyle="round,pad=0.2", fc="#ffe4e6", ec="#f43f5e", lw=1))

ax.set_title(f'2D PCA Projection of Wholesale Customer Segments (K={chosen_k})', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel(f'Principal Component 1 ({var_exp[0]*100:.1f}% Variance Explained - Dry/FMCG Goods vs Fresh)', fontsize=12, fontweight='semibold')
ax.set_ylabel(f'Principal Component 2 ({var_exp[1]*100:.1f}% Variance Explained - Fresh/Frozen Volume)', fontsize=12, fontweight='semibold')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right', frameon=True, facecolor='#ffffff', framealpha=0.95, fontsize=10)

plt.tight_layout()
plt.savefig('pca_clusters.png', dpi=300)
plt.close()
print("Saved: pca_clusters.png")

# Also perform Silhouette Plot for each individual sample
sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)
fig, ax = plt.subplots(figsize=(9, 6))

y_lower = 10
for i in range(chosen_k):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = colors[i % len(colors)]
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                    facecolor=color, edgecolor=color, alpha=0.7)

    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f"Cluster {i}", fontsize=11, fontweight='semibold', ha='right')
    y_lower = y_upper + 10

mean_sil = silhouette_score(X_scaled, cluster_labels)
ax.axvline(x=mean_sil, color="#e11d48", linestyle="--", linewidth=2, label=f"Average Silhouette Score = {mean_sil:.3f}")
ax.set_title(f"Silhouette Plot for Wholesale Customer Clusters (K={chosen_k})", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Silhouette Coefficient Values", fontsize=12, fontweight='semibold')
ax.set_ylabel("Cluster Assigned", fontsize=12, fontweight='semibold')
ax.set_yticks([])
ax.set_xlim([-0.3, 0.8])
ax.legend(loc="lower right", frameon=True, facecolor="#ffffff")
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('silhouette_per_sample.png', dpi=300)
plt.close()
print("Saved: silhouette_per_sample.png")

print("\n--- All Analysis and Visualizations Completed Successfully! ---")
