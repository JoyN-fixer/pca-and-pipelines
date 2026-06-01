# Integrating PCA in Pipelines

## Project Overview

This project demonstrates how to integrate **Principal Component Analysis (PCA)** into machine learning pipelines for classification tasks. Using the Otto Group Product Classification dataset, the project explores dimensionality reduction, model training, and hyperparameter optimization using several supervised learning algorithms.

The workflow follows a complete data science process:

1. Data inspection and exploratory data analysis (EDA)
2. Feature engineering using PCA
3. Baseline model development
4. Pipeline construction and model comparison
5. Hyperparameter tuning with GridSearchCV
6. Model evaluation and interpretation

---

## Dataset

The dataset comes from the Otto Group Product Classification Challenge and contains:

* **id**: Unique identifier for each product
* **feat_1 – feat_93**: Anonymous product features
* **target**: Product category label

### Key Findings

* No significant missing values were detected.
* Many features were heavily zero-inflated.
* Feature distributions were highly skewed.
* Correlations existed among several features, making dimensionality reduction beneficial.

---

## Feature Engineering with PCA

To reduce dimensionality while preserving information:

* Features were standardized using `StandardScaler`.
* PCA was applied to retain **80% of the explained variance**.
* The transformed dataset significantly reduced the number of dimensions while maintaining most of the predictive information.

## Baseline Model

A baseline pipeline was created using:

1. StandardScaler
2. PCA
3. Logistic Regression

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.80)),
    ('log', LogisticRegression(random_state=123))
])
```

---

## Model Comparison

Additional pipelines were developed and evaluated:

### Logistic Regression

* PCA + Logistic Regression

### Support Vector Machine (SVM)

* PCA + Linear SVM

### Decision Tree

* PCA + DecisionTreeClassifier

### Random Forest

* PCA + RandomForestClassifier

Models were compared using classification accuracy on the test set.

---

## Hyperparameter Tuning

### Random Forest + GridSearchCV

Parameters tuned:

* Number of estimators
* Maximum depth
* Minimum samples split
* Minimum samples leaf
* PCA explained variance

### AdaBoost + GridSearchCV

Parameters tuned:

* Number of estimators
* Learning rate
* PCA explained variance

### SVM + GridSearchCV (Optional)

Parameters tuned:

* Kernel type
* Regularization parameter (C)
* PCA explained variance

---

## Results

Grid Search optimization improved model performance compared to baseline models.

### Best Performing Model

The tuned **Support Vector Machine (SVM)** achieved the highest cross-validation score among the tested models.

Reasons for selecting SVM:

* Strong generalization performance
* Effective in high-dimensional spaces
* Consistent cross-validation results
* Improved classification accuracy after PCA transformation

---


## Key Learnings

This project demonstrates:

* How PCA reduces dimensionality while preserving information
* Building reusable machine learning pipelines
* Comparing multiple classification algorithms
* Using GridSearchCV for hyperparameter optimization
* Evaluating model performance using cross-validation

---

## Future Improvements

* Explore additional PCA variance thresholds
* Test XGBoost and Gradient Boosting models
* Perform feature importance analysis
* Implement stratified cross-validation
* Optimize SVM hyperparameters further

---

