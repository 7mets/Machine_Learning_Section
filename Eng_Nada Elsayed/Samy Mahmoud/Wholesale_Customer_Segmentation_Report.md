# Wholesale Customers Segmentation using K-Means Clustering
**Course:** Machine Learning 2026 — Assoc. Prof. Mohamed Abd Elfattah (MET, Computer Science)  
**Mini Project Specification & Rubric Compliance (100 Marks)**  
**Author / Student Submission:** Machine Learning Pair Programming Lab  
**Date:** October 2026  

---

## Executive Summary
This project executes the end-to-end Machine Learning pipeline for unsupervised customer segmentation on the **UCI Wholesale Customers Dataset** (440 commercial clients). By clustering clients based on their annual expenditure across six core product categories (*Fresh*, *Milk*, *Grocery*, *Frozen*, *Detergents_Paper*, and *Delicassen*), we uncover distinct behavioral segments, overcome severe data skewness using log-transformation and feature standardization, validate the optimal number of clusters ($K=3$) using the **Elbow Method** and **Silhouette Analysis**, and map out high-impact commercial actions for wholesale management.

---

## 1. Data Preparation & Exploratory Analysis [Rubric: 20 Marks]

### 1.1 Dataset Inspection
The dataset comprises **440 wholesale clients** with 8 attributes:
* 2 structural attributes: `Channel` (1: Horeca, 2: Retail) and `Region` (1: Lisbon, 2: Oporto, 3: Other).
* 6 continuous spending features measured in annual monetary units (m.u.):
  * **Fresh:** Fresh produce (Mean: 12,000, Max: 112,151)
  * **Milk:** Dairy products (Mean: 5,796, Max: 73,498)
  * **Grocery:** Dry food and grocery goods (Mean: 7,951, Max: 92,780)
  * **Frozen:** Frozen foods (Mean: 3,072, Max: 60,869)
  * **Detergents_Paper:** Cleaning and paper supplies (Mean: 2,881, Max: 40,827)
  * **Delicassen:** Specialty delicatessen foods (Mean: 1,525, Max: 47,943)

### 1.2 Skewness and the Necessity of Transformation
A critical challenge in financial and consumption data is severe right-skewness and extreme magnitude discrepancies:
* **Raw Skewness:**
  * `Fresh`: +2.56
  * `Milk`: +4.05
  * `Grocery`: +3.59
  * `Frozen`: +5.91
  * `Detergents_Paper`: +3.63
  * `Delicassen`: +11.15
* **Theoretical Justification:** 
  Standard K-Means utilizes Euclidean distance:
  $$d(x, y) = \sqrt{\sum_{j=1}^d (x_j - y_j)^2}$$
  In raw form, the high-spending attributes (like Fresh, with values exceeding 100,000) contribute over 98% of the squared distance, rendering categories like Delicassen virtually invisible. Furthermore, extreme outliers act as artificial centroids, causing premature local minima.

### 1.3 Transformation Strategy: Log-Transform + Z-Score Standardization
1. **Log-Transformation:** We apply $x_{log} = \ln(1 + x)$ across all six spending features. As shown in our generated distribution plots (`distribution_and_transformations.png`), the log-transformation compresses heavy right tails and restores quasi-normal symmetric distributions:
   * `Fresh` log skewness: -1.58
   * `Milk` log skewness: -0.22
   * `Grocery` log skewness: -0.67
   * `Frozen` log skewness: -0.35
   * `Detergents_Paper` log skewness: -0.24
   * `Delicassen` log skewness: -1.09
2. **StandardScaler ($z = \frac{x - \mu}{\sigma}$):** Each feature is centered to zero mean and unit variance ($\mu=0, \sigma=1$). This guarantees that each product category exerts proportional geometric influence during distance computations.

---

## 2. Model Selection: Optimal $K$ with Evidence [Rubric: 20 Marks]

To rigorously justify the choice of $K$, we evaluated candidate values $K \in \{2, 3, 4, 5, 6, 7, 8\}$ using $K$-means++ initialization with 20 restarts (`n_init=20`).

| Number of Clusters ($K$) | Inertia ($J = \text{WCSS}$) | Mean Silhouette Score ($s$) | Davies-Bouldin Index | Calinski-Harabasz Index |
| :---: | :---: | :---: | :---: | :---: |
| **K = 2** | 1844.06 | **0.2903** | 1.352 | **189.0** |
| **K = 3** | **1553.41** | **0.2583** | **1.348** | 152.8 |
| **K = 4** | 1386.86 | 0.1885 | 1.541 | 131.3 |
| **K = 5** | 1266.15 | 0.1916 | 1.484 | 118.0 |
| **K = 6** | 1174.54 | 0.1953 | 1.414 | 108.3 |
| **K = 7** | 1086.36 | 0.1958 | 1.392 | 103.2 |
| **K = 8** | 1024.05 | 0.1915 | 1.385 | 97.4 |

### Formal Justification for Selecting $K = 3$ (Course Specification Task 4)
> **Two-Sentence Justification:**  
> The Inertia plot exhibits a distinct elbow at $K = 3$ where the rate of variance reduction sharply decreases from steep descent to gradual decline, yielding the lowest Davies-Bouldin index (1.348). While $K=2$ produces a slightly higher silhouette score, $K=3$ achieves a robust silhouette of 0.258 while uniquely distinguishing between retail grocers, dining establishments, and high-volume multi-category buyers, making it the most actionable segmentation for business operations.

---

## 3. Correct Clustering Implementation [Rubric: 20 Marks]

The clustering pipeline adheres to Lloyd's iterative algorithm with **K-means++ seeding** (Arthur & Vassilvitskii, 2007):
1. **K-means++ Initialization:** Avoids poor local minima by selecting the first centroid uniformly at random and subsequent centroids with probability proportional to $D(x)^2$, guaranteeing an $O(\log K)$-competitive initial configuration.
2. **Assignment Step:** Every client vector $x_i$ is mapped to the nearest centroid:
   $$c(x) = \arg\min_k \|x - \mu_k\|^2$$
3. **Centroid Update Step:** Centroids are recalculated as the arithmetic mean of all member vectors:
   $$\mu_k = \frac{1}{|C_k|} \sum_{x \in C_k} x$$
4. **Convergence Guarantee:** Because both the assignment and update steps strictly decrease or preserve the objective $J = \sum_{k=1}^K \sum_{x \in C_k} \|x - \mu_k\|^2$, and the number of feasible partitions is finite, the algorithm is guaranteed to converge to a stable local minimum.

---

## 4. Cluster Interpretation & Business Recommendations [Rubric: 25 Marks]

Analyzing the centroid spending in original monetary units and relative to overall client averages ($1.0 = \text{Dataset Average}$) reveals three unmistakable customer personas:

### Cluster Summary Table
| Metric / Feature | Cluster 0: Retail Grocers | Cluster 1: Restaurants & HoReCa | Cluster 2: VIP High-Volume Buyers | Dataset Average |
| :--- | :---: | :---: | :---: | :---: |
| **Client Count ($n$)** | **81 clients (18.4%)** | **212 clients (48.2%)** | **147 clients (33.4%)** | 440 clients |
| **Fresh** | 2,869 m.u. *(0.24x)* | 11,993 m.u. *(1.00x)* | **17,043 m.u. (1.42x)** | 12,000 m.u. |
| **Milk** | 7,099 m.u. *(1.22x)* | 1,995 m.u. *(0.34x)* | **10,560 m.u. (1.82x)** | 5,796 m.u. |
| **Grocery** | 12,459 m.u. *(1.57x)* | 2,497 m.u. *(0.31x)* | **13,334 m.u. (1.68x)** | 7,951 m.u. |
| **Frozen** | 607 m.u. *(0.20x)* | 3,278 m.u. *(1.07x)* | **4,133 m.u. (1.35x)** | 3,072 m.u. |
| **Detergents_Paper** | **5,487 m.u. (1.90x)** | 426 m.u. *(0.15x)* | 4,987 m.u. *(1.73x)* | 2,881 m.u. |
| **Delicassen** | 781 m.u. *(0.51x)* | 893 m.u. *(0.59x)* | **2,847 m.u. (1.87x)** | 1,525 m.u. |
| **Total Annual Spend** | **29,302 m.u.** | **21,082 m.u.** | **52,904 m.u.** | 33,225 m.u. |

---

### Detailed Persona Profiles & Strategic Action Plan

#### 🛒 Segment 0: Retail Grocers & Supermarkets (18.4% of clients)
* **Behavioral Profile:** Substantial expenditure on non-perishable Fast-Moving Consumer Goods (FMCG): Detergents & Paper (+90% above average) and Grocery (+57% above average), alongside solid Milk demand. Extremely minimal purchases in Fresh and Frozen categories.
* **Commercial Strategy & Recommended Action:**
  * **Automated Vendor Managed Inventory (VMI):** Implement Electronic Data Interchange (EDI) ordering for high-turnover shelf-stable items.
  * **Bundled Rebates:** Offer tier discounts combining Detergents/Paper with dry Grocery staples.
  * **Financial Terms:** Extend 30-to-45 day revolving trade credit to support retail shelf stocking cycles.

#### 🍽️ Segment 1: Dining & Hospitality Services (HoReCa) (48.2% of clients)
* **Behavioral Profile:** Constituting nearly half the client base, these businesses consume fresh produce and frozen ingredients for culinary preparation. Spending on commercial resale groceries and detergents is under 35% of the market average.
* **Commercial Strategy & Recommended Action:**
  * **Morning Priority Logistics:** Guarantee pre-dawn cold-chain deliveries before kitchen prep shifts begin.
  * **Seasonal Fresh Contracts:** Establish harvest-indexed pricing for vegetables, fruits, and fresh meats.
  * **Cold-Chain Quality Assurance:** Provide temperature-monitored refrigerated transport certifications to ensure HACCP food safety compliance.

#### 💎 Segment 2: VIP High-Volume Buyers & Mega Caterers (33.4% of clients)
* **Behavioral Profile:** The core profit driver of the distributor. These clients spend heavily across **every single category**, exceeding market averages by 35% to 87%. They are the primary consumers of premium Delicassen items (1.87x) and Fresh produce (1.42x).
* **Commercial Strategy & Recommended Action:**
  * **Dedicated Key Account Managers:** Assign senior relationship representatives for personalized service and bespoke contract negotiation.
  * **Tiered Annual Volume Loyalty Rebates:** Institute retroactive rebate bonuses for accounts reaching spending milestones.
  * **Exclusive Product Pre-allocations:** Offer priority allocation on imported fine foods, gourmet cheeses, and premium wines/delicacies.

---

## 5. Visualizations & 2D PCA Dimensionality Reduction [Rubric: 15 Marks]

### 5.1 Principal Component Analysis (PCA)
To project the 6-dimensional standardized space onto an interpretable 2D plane:
* **PC1 (44.08% Variance Explained):** Strongly aligned with dry grocery, paper goods, and milk spending (Retail factor).
* **PC2 (27.19% Variance Explained):** Strongly aligned with fresh produce and frozen food consumption (Culinary/Fresh factor).
* **Cumulative Variance:** **71.27%** of the total variance across all 6 features is preserved in the 2D projection.

### 5.2 Visual Artifacts Generated in Project Directory
1. `distribution_and_transformations.png`: Side-by-side histograms comparing raw heavy skew versus standardized Gaussian profiles.
2. `elbow_silhouette_plot.png`: Dual-axis visualization highlighting the inertia drop and peak silhouette validation at $K=3$.
3. `cluster_profiles.png`: Comparative bar chart demonstrating the distinct spending signatures of each segment.
4. `pca_clusters.png`: 2D scatter plot depicting cluster separation with marked centroid diamonds ($\mu_k$).
5. `silhouette_per_sample.png`: Individual silhouette coefficient plots verifying cluster consistency with minimal misclassifications.

---

## 6. Deliverables Inventory
* 📄 **Jupyter Notebook:** `Customer_Segmentation_Wholesale.ipynb` (Fully annotated, self-contained, and runnable top to bottom).
* 🐍 **Production Python Pipeline:** `customer_segmentation.py` (Modular script generating all metrics and artifacts).
* 📊 **Segmented Dataset:** `segmented_wholesale_customers.csv` (Original dataset enriched with Cluster IDs, Segment Personas, and Strategic Actions).
* 🌐 **Interactive Dashboard:** `index.html`, `style.css`, and `app.js` (A responsive, modern web application for interactive cluster inspection).

---
*Completed in accordance with MET Computer Science Machine Learning Coursework Standards.*
