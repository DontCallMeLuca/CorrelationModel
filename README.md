<h1 align="center">✨ Correlation Model ✨</h1>

<h6 align="center"><em>Simple machine learning correlation model</em></h6>

# Overview
This project implements a `Random Forest Regression` machine learning model.

## Features
- Automated data scaling
- Automated train-test splitting
- Feature standardization using `StandardScaler`
- Random Forest Regression with **1000** estimators
- Accuracy evaluation via **MAPE**

# Mathematical Foundation

## Data Preprocessing

### Standard Scaling
For each feature $x_i$ in the input dataset, the scaling transformation is:

```math
\begin{alignat*}{2}
& z_i = \frac{x_i - \mu_i}{\sigma_i}
\end{alignat*}
```

where:
- $z_i$ is the scaled feature
- $\mu_i$ is the mean of the feature
- $\sigma_i$ is the standard deviation of the feature

## Random Forest Regression

### Ensemble Structure
The model consists of $n = 1000$ decision trees, where each tree $T_j$ contributes to the final prediction:

```math
\begin{alignat*}{2}
& f(x) = \frac{1}{n}\sum_{j=1}^n T_j(x)
\end{alignat*}
```

### Tree Construction
Each tree is built by:

1. **Bootstrap Sampling**: For each tree $j$, create a bootstrap sample $D_j$ from the training data $D$

2. **Split Selection**: At each node $k$, select the best split $s^*$ that maximizes the variance reduction:

```math
\begin{alignat*}{2}
& s^* = \argmax_{s} \left[\text{Var}(Y_p) - \left(\frac{n_l}{n_p}\text{Var}(Y_l) + \frac{n_r}{n_p}\text{Var}(Y_r)\right)\right]
\end{alignat*}
```

where:
- $Y_p$ is the set of target values at parent node
- $Y_l, Y_r$ are the sets of target values in left and right child nodes
- $n_p, n_l, n_r$ are the number of samples in parent and child nodes

## Model Evaluation

### Mean Absolute Percentage Error (MAPE)
For predictions $\hat{y}_i$ and true values $y_i$:

```math
\begin{alignat*}{2}
& \text{MAPE} = \frac{100}{n}\sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right|
\end{alignat*}
```

### Model Accuracy
The accuracy is derived from MAPE:

```math
\begin{alignat*}{2}
& \text{Accuracy} = 100 - \text{MAPE}
\end{alignat*}
```

## Prediction Process

1. **Input Scaling**: New input $x_{new}$ is scaled using stored $\mu$ and $\sigma$:

```math
\begin{alignat*}{2}
& z_{new} = \frac{x_{new} - \mu}{\sigma}
\end{alignat*}
```

2. **Ensemble Prediction**: Each tree makes a prediction, and the final prediction is their average:

```math
\begin{alignat*}{2}
& \hat{y} = \frac{1}{n}\sum_{j=1}^n T_j(z_{new})
\end{alignat*}
```

## Key Properties

1. **Variance Reduction**: Through averaging $n$ trees:
```math
\begin{alignat*}{2}
& \text{Var}(\hat{y}) = \frac{\rho\sigma^2 + (1-\rho)\sigma^2/n}{\sigma^2}
\end{alignat*}
```
where $\rho$ is the correlation between trees and $\sigma^2$ is the variance of a single tree.

2. **Training-Test Split**: 85-15 split ratio:
```math
\begin{alignat*}{2}
& |D_{train}| = 0.85|D|\\
& |D_{test}| = 0.15|D|
\end{alignat*}
```

# Usage

```python
# Create instance with your dataset
model = CorrelationModel(your_dataframe, 'Target_Feature')

# Train the model and get accuracy metrics
X_test, y_test, accuracy, mape = model._train_model()

# Make predictions
predictions = model.predict(new_data)
```

# Input Data Format
The model expects a pandas DataFrame object with:
- Target variable
- Feature columns (any number)

**Example:**
```python
data = pd.DataFrame({
    'feature1': [...],
    'feature2': [...]
})
```

**Model Parameters**
1. Random Forest:
- `n_estimators: 1000`
- `random_state: 42`
2. Train-Test Split:
- Default test size: **15%**

# Properties
- `dataset`: Access the full dataset
- `target_label`: Access the target vector's label
- `X`: Feature matrix (excludes `y` vector)
- `y`: Target variable
- `model`: Trained model instance
- `scaler`: StandardScaler instance

# Notes

Please note I made this model a long time ago,
it was originally made for price prediction based on features.

# License
This project uses the `GNU GENERAL PUBLIC LICENSE v3.0` license
<br>
For more info, please find the `LICENSE` file here: [License](LICENSE)