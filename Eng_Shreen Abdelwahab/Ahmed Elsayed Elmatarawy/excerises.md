# دليل الإجابات المباشرة الشامل لمادة تعليم الآلة (Machine Learning)

---

## المحاضرة الأولى: Introduction to Machine Learning

### أولاً: أسئلة الاختيار من متعدد (MCQs)

1. **Question:** Which component of machine learning defines what "good" looks like by scoring how well the model is performing?
   * **Answer:** C) Loss function

2. **Question:** A bank wants to flag credit card transactions as fraudulent or legitimate based on past labeled data. Which type of ML is this?
   * **Answer:** B) Supervised classification

3. **Question:** In the train-validation-test split, what is the role of the test set?
   * **Answer:** C) Final unbiased estimate of performance

4. **Question:** A model scores 99% accuracy on the training set but only 58% on new test data. What is happening?
   * **Answer:** B) Overfitting (high variance)

5. **Question:** Why should continuous features be scaled (e.g. z-score) before distance-based ML algorithms?
   * **Answer:** B) Large-scale features dominate distance metrics wrongly

---

### ثانياً: الأسئلة المقالية والأفهومات

1. **Question:** Explain the difference between Supervised, Unsupervised, and Reinforcement Learning with one concrete example each.
   * **Answer:** 
     * **Supervised Learning:** Learns from labeled data $(x, y)$. *Example:* Email spam classification.
     * **Unsupervised Learning:** Finds patterns in unlabeled data $(x)$. *Example:* Customer segmentation via clustering.
     * **Reinforcement Learning:** Learns by interacting with an environment through rewards and penalties. *Example:* Chess-playing AI.

2. **Question:** Differentiate between Overfitting and Underfitting. Mention one technique to solve each.
   * **Answer:** 
     * **Overfitting:** Low training error, high test error (High Variance). *Solution:* Regularization (L1/L2) or reducing model complexity.
     * **Underfitting:** High training error, high test error (High Bias). *Solution:* Increasing model complexity or adding features.

3. **Question:** What is the purpose of Data Normalization / Feature Scaling?
   * **Answer:** To bring all features to a similar scale so that distance-based algorithms and gradient descent converge faster and don't get dominated by features with larger numerical ranges.

---

### ثالثاً: التمارين والمسائل العددية (Numerical Problems)

1. **Question:** Given the following target values $y = [2, 4, 5, 8, 10]$ and predictions $\hat{y} = [2.5, 3.5, 6, 7, 11]$, calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE).
   * **Answer:** 
     * $\text{MSE} = 0.55$
     * $\text{MAE} = 0.7$

2. **Question:** Given a dataset with $N = 100$ samples, standard scaling is applied to a feature $X$ with $\mu = 50$ and $\sigma = 10$. Find the transformed value $z$ for $x = 65$.
   * **Answer:** $z = 1.5$

---

## المحاضرة الثانية: Linear & Logistic Regression

### أولاً: أسئلة الاختيار من متعدد (MCQs)

1. **Question:** What loss function is standard for evaluating Linear Regression?
   * **Answer:** A) Mean Squared Error (MSE)

2. **Question:** In Logistic Regression, what function maps the linear output to a probability between 0 and 1?
   * **Answer:** B) Sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$

3. **Question:** What effect does L2 regularization (Ridge) have on model parameters?
   * **Answer:** C) Shrinks weights toward zero but rarely makes them exactly zero

4. **Question:** What loss function is used to train Logistic Regression models?
   * **Answer:** B) Binary Cross-Entropy (Log Loss)

5. **Question:** What does gradient descent do during training?
   * **Answer:** C) Updates weights in the opposite direction of the gradient to minimize loss

---

### ثانياً: الأسئلة المقالية والأفهومات

1. **Question:** Why is Mean Squared Error (MSE) NOT suitable as a loss function for Logistic Regression?
   * **Answer:** Because combining MSE with the non-linear Sigmoid function results in a non-convex loss function with multiple local minima, making gradient descent fail to find the global minimum.

2. **Question:** What is the main difference between L1 (Lasso) and L2 (Ridge) Regularization?
   * **Answer:** L1 regularization forces some weights to become exactly zero (performing feature selection), whereas L2 regularization shrinks weights near zero without setting them to exact zero.

---

### ثالثاً: التمارين والمسائل العددية (Numerical Problems)

1. **Question:** Given a simple linear model $\hat{y} = 2x + 1$, learning rate $\alpha = 0.1$, training example $(x=3, y=8)$, and current loss $L = \frac{1}{2}(\hat{y} - y)^2$: Calculate the updated weight $w$ after 1 step of Gradient Descent.
   * **Answer:** $w^{(new)} = 1.7$

2. **Question:** For a logistic regression model with weight $w = 0.5$ and bias $b = -1$, calculate the predicted probability for $x = 4$.
   * **Answer:** $P(y=1|x) = 0.731$ (or $73.1\%$)

---

## المحاضرة الثالثة: Decision Trees & Random Forests

### أولاً: أسئلة الاختيار من متعدد (MCQs)

1. **Question:** Which metric measures impurity in a Decision Tree classification node?
   * **Answer:** C) Gini Impurity / Entropy

2. **Question:** What is the primary technique used in Random Forests to reduce variance?
   * **Answer:** B) Bagging (Bootstrap Aggregating) and random feature selection

3. **Question:** A decision tree of unlimited depth is most likely to suffer from:
   * **Answer:** A) Overfitting

4. **Question:** What is the minimum possible Gini Impurity for a completely pure node?
   * **Answer:** A) 0.0

---

### ثانياً: الأسئلة المقالية والأفهومات

1. **Question:** Explain how Decision Trees handle missing values and feature scaling.
   * **Answer:** Decision Trees are non-parametric models; they are completely invariant to feature scaling and can handle missing values by splitting rules or surrogate splits.

2. **Question:** How does Random Forest improve upon a single Decision Tree?
   * **Answer:** By building an ensemble of decorrelated decision trees using bootstrap sampling and random sub-feature selection, which significantly reduces variance without increasing bias.

---

### ثالثاً: التمارين والمسائل العددية (Numerical Problems)

1. **Question:** Calculate the Gini Impurity for a node containing 6 positive classes and 4 negative classes.
   * **Answer:** $\text{Gini} = 0.48$

2. **Question:** Calculate the Entropy for a dataset node with 8 Positive samples and 8 Negative samples.
   * **Answer:** $\text{Entropy} = 1.0 \text{ bit}$

---

## المحاضرة الرابعة: Support Vector Machines (SVM) & k-NN

### أولاً: أسئلة الاختيار من متعدد (MCQs)

1. **Question:** What does SVM try to maximize?
   * **Answer:** B) Margin between decision boundary and support vectors

2. **Question:** What is the purpose of the Kernel Trick in SVM?
   * **Answer:** C) Map data into higher-dimensional space to make it linearly separable without computing high-dim coordinates explicitly

3. **Question:** In k-Nearest Neighbors (k-NN), setting $k=1$ typically leads to:
   * **Answer:** A) High variance (Overfitting)

4. **Question:** Which hyperparameter in SVM controls the trade-off between maximizing margin and minimizing training errors?
   * **Answer:** B) $C$ parameter

---

### ثانياً: الأسئلة المقالية والأفهومات

1. **Question:** What are Support Vectors in an SVM model?
   * **Answer:** Support vectors are the training data points that lie closest to the decision boundary (margin boundaries) and directly define the hyper-plane position and orientation.

2. **Question:** Why is k-NN referred to as a "Lazy Learner"?
   * **Answer:** Because it does not learn an explicit discriminative model during training; instead, it stores the dataset and performs computation only during test/prediction time.

---

### ثالثاً: التمارين والمسائل العددية (Numerical Problems)

1. **Question:** Given target point $P=(2, 3)$, and neighbor points $A=(1, 1) [y=+1]$, $B=(4, 5) [y=-1]$, $C=(2, 5) [y=+1]$. Determine the prediction using 3-NN under Euclidean distance.
   * **Answer:** Class $+1$ (Distances: $A=\sqrt{5}\approx 2.23, B=\sqrt{13}\approx 3.60, C=2.0$. The 3 nearest are $C, A, B$, majority vote gives $+1$).

---

## المحاضرة الخامسة: Unsupervised Learning (Clustering & PCA)

### أولاً: أسئلة الاختيار من متعدد (MCQs)

1. **Question:** K-Means clustering algorithm aims to minimize which objective function?
   * **Answer:** B) Within-Cluster Sum of Squares (WCSS / Inertia)

2. **Question:** Principal Component Analysis (PCA) is primarily used for:
   * **Answer:** A) Dimensionality reduction while maximizing variance retention

3. **Question:** Which clustering algorithm does NOT require specifying the number of clusters $k$ beforehand?
   * **Answer:** C) DBSCAN / Hierarchical Clustering

4. **Question:** In PCA, the principal components correspond to the:
   * **Answer:** B) Eigenvectors of the data covariance matrix

---

### ثانياً: الأسئلة المقالية والأفهومات

1. **Question:** Briefly explain the steps of the K-Means algorithm.
   * **Answer:** 
     1. Initialize $k$ cluster centroids randomly.
     2. Assign each data point to the nearest centroid.
     3. Recalculate centroids as the mean of assigned points.
     4. Repeat steps 2-3 until centroids stop changing (convergence).

2. **Question:** What is the Elbow Method used for in K-Means?
   * **Answer:** It is a heuristic method used to determine the optimal number of clusters ($k$) by plotting WCSS vs. $k$ and finding the point where the rate of decrease sharply shifts (the "elbow").

---

### ثالثاً: التمارين والمسائل العددية (Numerical Problems)

1. **Question:** Given data points $X = [1, 2, 5, 6]$ and initial centroids $c_1 = 1, c_2 = 4$. Perform 1 iteration of K-Means and find new centroids.
   * **Answer:** 
     * Cluster 1 points: $\{1, 2\} \rightarrow \text{New } c_1 = 1.5$
     * Cluster 2 points: $\{5, 6\} \rightarrow \text{New } c_2 = 5.5$

2. **Question:** A 2D covariance matrix has eigenvalues $\lambda_1 = 8$ and $\lambda_2 = 2$. What percentage of total variance is explained by the first principal component?
   * **Answer:** $80\%$

---

## المحاضرة السادسة: Model Evaluation & Hyperparameter Tuning

### أولاً: أسئلة الاختيار من متعدد (MCQs)

1. **Question:** Which metric should be prioritized for highly imbalanced datasets where detecting positives is crucial (e.g. disease detection)?
   * **Answer:** B) Recall (Sensitivity)

2. **Question:** What does the Area Under the ROC Curve (ROC-AUC) score measure?
   * **Answer:** B) Model's capability to distinguish between positive and negative classes across all thresholds

3. **Question:** What is the main advantage of K-Fold Cross-Validation over a single Train/Test split?
   * **Answer:** C) Provides a more robust evaluation with lower variance estimate

4. **Question:** F1-Score is defined as the harmonic mean of:
   * **Answer:** A) Precision and Recall

---

### ثانياً: الأسئلة المقالية والأفهومات

1. **Question:** Define Precision and Recall. Write their mathematical formulas.
   * **Answer:** 
     * **Precision:** Ratio of true positives among all predicted positives. $\text{Precision} = \frac{TP}{TP + FP}$
     * **Recall:** Ratio of true positives identified out of all actual positives. $\text{Recall} = \frac{TP}{TP + FN}$

2. **Question:** Explain the difference between Grid Search and Random Search for hyperparameter tuning.
   * **Answer:** Grid Search evaluates every possible combination in a specified grid exhaustively, whereas Random Search randomly samples combinations from hyperparameter distributions, usually finding optimal solutions much faster.

---

### ثالثاً: التمارين والمسائل العددية (Numerical Problems)

1. **Question:** Given a confusion matrix: $TP = 70, FP = 10, FN = 20, TN = 900$. Calculate Precision, Recall, Accuracy, and F1-Score.
   * **Answer:** 
     * $\text{Accuracy} = 97\%$ ($0.97$)
     * $\text{Precision} = 87.5\%$ ($0.875$)
     * $\text{Recall} = 77.78\%$ ($0.7778$)
     * $\text{F1-Score} = 82.35\%$ ($0.8235$)