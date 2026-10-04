"""Linear Regression from scratch (NumPy only) using Gradient Descent.

Run:  python linear_regression.py      (needs Housing.csv in the same folder)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. The algorithm
# ============================================================

def cost_function(x, y, w, b):
    """J(w,b) = 1/(2m) * sum((x.w + b - y)^2)"""
    m = len(x)
    errors = np.dot(x, w) + b - y
    return (1 / (2 * m)) * np.sum(errors ** 2)


def gradient_function(x, y, w, b):
    """dJ/dw = 1/m * x^T . error      dJ/db = 1/m * sum(error)"""
    m = len(x)
    error = np.dot(x, w) + b - y
    d_dw = (1 / m) * np.dot(x.T, error)
    d_db = (1 / m) * np.sum(error)
    return d_dw, d_db


def gradient_descent(x, y, alpha, iterations):
    """Repeat: move w and b a small step (alpha) against the gradient."""
    w = np.zeros(x.shape[1])
    b = 0.0
    cost_history = []
    for i in range(iterations):
        d_dw, d_db = gradient_function(x, y, w, b)
        w = w - alpha * d_dw
        b = b - alpha * d_db
        cost_history.append(cost_function(x, y, w, b))
        if i % 500 == 0:
            print(f"Iteration {i:5d}: Cost = {cost_history[-1]:.6f}")
    return w, b, cost_history


# ============================================================
# 2. Load and prepare the data
# ============================================================

df = pd.read_csv("Housing.csv").dropna()
y = df["price"].to_numpy(dtype=float)
X_df = df.drop(columns=["price"])

for col in X_df.select_dtypes(exclude="number").columns:     # yes/no -> 1/0
    if set(X_df[col].str.lower().unique()) <= {"yes", "no"}:
        X_df[col] = (X_df[col].str.lower() == "yes").astype(int)
X_df = pd.get_dummies(X_df, drop_first=True, dtype=float)    # other text -> one-hot
names = X_df.columns.tolist()
X = X_df.to_numpy(dtype=float)

# Train / test split (80/20, fixed seed so results are repeatable)
idx = np.random.default_rng(42).permutation(len(X))
n_test = int(0.2 * len(X))
X_test, y_test = X[idx[:n_test]], y[idx[:n_test]]
X_train, y_train = X[idx[n_test:]], y[idx[n_test:]]

# Standardize with TRAIN statistics only (so the test set stays unseen)
X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
X_std[X_std == 0] = 1.0
y_mean, y_std = y_train.mean(), y_train.std()
X_norm = (X_train - X_mean) / X_std
y_norm = (y_train - y_mean) / y_std


# ============================================================
# 3. Train
# ============================================================

w_norm, b_norm, cost_history = gradient_descent(X_norm, y_norm, alpha=0.05, iterations=5000)

# Convert the weights back to the original units (so we can read them)
w = (y_std / X_std) * w_norm
b = y_mean + y_std * b_norm - np.dot(X_mean / X_std, w_norm) * y_std


# ============================================================
# 4. Evaluate on unseen data and compare with other methods
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2_score(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)

pred = np.dot(X_test, w) + b

# Normal Equation = exact closed-form answer (gradient descent should match it)
A = np.c_[X_train, np.ones(len(X_train))]
theta = np.linalg.lstsq(A, y_train, rcond=None)[0]
pred_ne = np.dot(X_test, theta[:-1]) + theta[-1]

print(f"\n{'Method':<24}{'RMSE':>14}{'R2':>9}")
print(f"{'Gradient Descent (mine)':<24}{rmse(y_test, pred):>14,.0f}{r2_score(y_test, pred):>9.4f}")
print(f"{'Normal Equation':<24}{rmse(y_test, pred_ne):>14,.0f}{r2_score(y_test, pred_ne):>9.4f}")
try:
    from sklearn.linear_model import LinearRegression
    pred_sk = LinearRegression().fit(X_train, y_train).predict(X_test)
    print(f"{'sklearn':<24}{rmse(y_test, pred_sk):>14,.0f}{r2_score(y_test, pred_sk):>9.4f}")
except ImportError:
    pass

print("\nLearned weights (original units):")
for n, wi in zip(names, w):
    print(f"  {n:<30}{wi:>14,.2f}")
print(f"  {'intercept':<30}{b:>14,.2f}")


# ============================================================
# 5. Plots
# ============================================================

os.makedirs("images", exist_ok=True)

plt.figure(figsize=(6, 4))
plt.plot(cost_history)
plt.xlabel("Iteration"); plt.ylabel("Cost"); plt.title("Loss curve")
plt.tight_layout(); plt.savefig("images/loss_curve.png", dpi=150); plt.close()

plt.figure(figsize=(5, 5))
plt.scatter(y_test, pred, alpha=0.6)
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
plt.plot(lims, lims, "r--", label="perfect prediction")
plt.xlabel("Actual price"); plt.ylabel("Predicted price"); plt.title("Predicted vs Actual (test set)")
plt.legend(); plt.tight_layout(); plt.savefig("images/pred_vs_actual.png", dpi=150); plt.close()
print("\nPlots saved in ./images")


# ============================================================
# 6. Predict a new house
# ============================================================
# Only fill what you want; any column you leave out is set to 0.
# yes/no columns: 1 = yes, 0 = no.  Example text columns: "furnishingstatus_semi-furnished": 1

new_house = {
    "area": 6000,
    "bedrooms": 3,
    "bathrooms": 2,
    "stories": 2,
    "parking": 1,
    "airconditioning": 1,
}
x_new = pd.Series(new_house).reindex(names, fill_value=0).to_numpy(dtype=float)
print(f"\nPredicted price for the new house: {np.dot(x_new, w) + b:,.0f}")
