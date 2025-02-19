<h1 align="center">🔎 Correlation Model 🔎</h1>

<h6 align="center"><em>Simple machine learning correlation model</em></h6>

## Overview
This project implements a `Random Forest Regression` machine learning model.
<br>
It's currently specifically applied for price prediction in euros.
<br>
But with some simple alterations this can change.

### Features
- Automated data scaling
- Automated train-test splitting
- Feature standardization using `StandardScaler`
- Random Forest Regression with **1000** estimators
- Accuracy evaluation via **MAPE**

## Mathematical Foundation

### Standard Scaling

The model uses StandardScaler for feature normalization:
```math
z = \frac{x - \mu}{\sigma}
```
**Where:**
- `z` is the scaled value
- `x` is the original value
- `μ` is the mean of the feature
- `σ` is the standard deviation

### Model Accuracy
The model's accuracy is calculated using Mean Absolute Percentage Error (**MAPE**):
```math
MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left|\frac{A_i - P_i}{A_i}\right|
```

**Where:**
- `n` is the number of observations
- `A_i` is the actual value
- `P_i` is the predicted value

The final accuracy is calculated as:
```math
Accuracy = 100\% - MAPE
```

## Usage

```python
# Create instance with your dataset
model = CorrelationModel(your_dataframe)

# Train the model and get accuracy metrics
X_test, y_test, accuracy, mape = model._train_model()

# Make predictions
predictions = model.predict(new_data)
```

## Input Data Format
The model expects a pandas DataFrame object with:
- Target variable named `Price_euros`
- Feature columns (any number)

**Example:**
```python
data = pd.DataFrame({
    'feature1': [...],
    'feature2': [...],
    'Price_euros': [...]
})
```

**Model Parameters**
1. Random Forest:
- `n_estimators: 1000`
- `random_state: 42`
2. Train-Test Split:
- Default test size: **15%**

## Properties
- `dataset`: Access the full dataset
- `X`: Feature matrix (excludes `y` vector)
- `y`: Target variable (`Price_euros`)
- `model`: Trained model instance
- `scaler`: StandardScaler instance

## Notes

Please note I made this model a long time ago,
it was originally made for price prediction based on features.
<br>
You can easily adapt this to your own needs. As with all models,
the more data you give it, the better.

## License
This project uses the `GNU GENERAL PUBLIC LICENSE v3.0` license
<br>
For more info, please find the `LICENSE` file here: [License](LICENSE)