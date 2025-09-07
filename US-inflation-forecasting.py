import os 
import pandas as pd
import numpy as np
from fredapi import Fred
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_fred() -> Fred:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not set. Export it in your shell: "
            'export FRED_API_KEY="YOUR_REAL_KEY"'
        )
    return Fred(api_key=api_key)
fred = get_fred()

def to_month_end(s: pd.Series) -> pd.Series:
    """Coerce to month-end timestamps and monthly frequency."""
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    s.index = s.index.to_period("M").to_timestamp("M")
    return s.asfreq("M")

def quarterly_to_monthly_ffill(s: pd.Series) -> pd.Series:
    """Quarterly series -> month-end, monthly freq by forward-filling within the quarter."""
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    s.index = s.index.to_period("Q").to_timestamp("M", "end")
    s = s.asfreq("M").ffill()
    s.index = s.index.to_period("M").to_timestamp("M")
    return s

FRED_SERIES = {
    "CPI": "CPIAUCSL",
    "UNRATE": "UNRATE",
    "FEDFUNDS": "FEDFUNDS",
    "WTI": "MCOILWTICO",
    "RGDP": "GDPC1",  
}

cpi     = to_month_end(fred.get_series(FRED_SERIES["CPI"]))
unrate  = to_month_end(fred.get_series(FRED_SERIES["UNRATE"]))
fedfunds= to_month_end(fred.get_series(FRED_SERIES["FEDFUNDS"]))
wti     = to_month_end(fred.get_series(FRED_SERIES["WTI"]))
rgdp_q  = fred.get_series(FRED_SERIES["RGDP"])
rgdp    = quarterly_to_monthly_ffill(rgdp_q)


series = [cpi, unrate, fedfunds, wti, rgdp]
start  = max(s.dropna().index.min() for s in series)
end    = min(s.dropna().index.max() for s in series)
aligned = [s.loc[start:end] for s in series]

df = pd.concat(aligned, axis=1)
df.columns = ["CPI", "UNRATE", "FEDFUNDS", "WTI", "RGDP"]


def inflation_mom(cpi: pd.Series) -> pd.Series:
    return 100 * (np.log(cpi) - np.log(cpi.shift(1)))

df["inf_mom"] = inflation_mom(df["CPI"])
df["target"]  = df["inf_mom"].shift(-1)  


def add_lags(df: pd.DataFrame, cols, lags):
    df = df.copy()
    for c in cols:
        for L in lags:
            df[f"{c}_l{L}"] = df[c].shift(L)
    return df

lag_cols = ["inf_mom", "UNRATE", "FEDFUNDS", "WTI"]
df = add_lags(df, lag_cols, lags=[1])

lag_feature_cols = [
    c for c in df.columns
    if any(c.endswith(suf) for suf in ["_l1"])   
]
features = lag_feature_cols

XY = df[features + ["target"]].dropna()
X = XY[features]
y = XY["target"]

TEST_WINDOW_MONTHS = 36  # you can change to 24 or 48 later

def chronosplit(frame: pd.DataFrame, test_months: int):
    cutoff = frame.index.max() - pd.offsets.MonthEnd(test_months)
    train = frame.loc[:cutoff]
    test  = frame.loc[cutoff + pd.offsets.MonthEnd(0):]
    return train, test

train_df, test_df = chronosplit(XY, TEST_WINDOW_MONTHS)
X_train, y_train = train_df[features], train_df["target"]
X_test,  y_test  = test_df[features],  test_df["target"]

print("Train/Test sizes:", X_train.shape, X_test.shape)

y_pred_naive = X_test["inf_mom_l1"].values
baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_naive))
baseline_mae  = mean_absolute_error(y_test, y_pred_naive)
print(f"\nBaseline  RMSE={baseline_rmse:.3f}  MAE={baseline_mae:.3f}")

tscv = TimeSeriesSplit(n_splits=5)

# OLS
ols_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ols", LinearRegression())
])
ols_pipe.fit(X_train, y_train)
pred_ols   = ols_pipe.predict(X_test)
ols_rmse   = np.sqrt(mean_squared_error(y_test, pred_ols))
ols_mae    = mean_absolute_error(y_test, pred_ols)
print(f"OLS       RMSE={ols_rmse:.3f}  MAE={ols_mae:.3f}")

ridge_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])
alphas = np.logspace(-6, 6, 25)
grid = GridSearchCV(
    ridge_pipe,
    param_grid={"ridge__alpha": alphas},
    scoring="neg_mean_squared_error",
    cv=tscv,
    n_jobs=-1
)
grid.fit(X_train, y_train)
pred_ridge = grid.best_estimator_.predict(X_test)
ridge_rmse = np.sqrt(mean_squared_error(y_test, pred_ridge))
ridge_mae  = mean_absolute_error(y_test, pred_ridge)
print(f"Ridge(a={grid.best_params_['ridge__alpha']:.2e})  RMSE={ridge_rmse:.3f}  MAE={ridge_mae:.3f}")

df["inf_mom"] = inflation_mom(df["CPI"])
df["target"]  = df["inf_mom"].shift(-1)  # predict next month's inflation

df["month"] = df.index.month
month_dummies = pd.get_dummies(df["month"], prefix="m", drop_first=True)  
df = pd.concat([df, month_dummies], axis=1)

# 1) Start from lag features
lag_feature_cols = [
    c for c in df.columns
    if any(c.endswith(suf) for suf in ["_l1","_l3","_l6","_l12"])
]

seasonality_cols = [c for c in df.columns if c.startswith("m_")]

# 3) Final feature list
features = lag_feature_cols + seasonality_cols

# 4) Build modeling table (drop rows with NaNs from lags)
XY = df[features + ["target"]].dropna()
X = XY[features]
y = XY["target"]


# --- chronological split: last N months as test ---
TEST_WINDOW_MONTHS = 36  # you can change to 24 or 48 later

def chronosplit(frame: pd.DataFrame, test_months: int):
    cutoff = frame.index.max() - pd.offsets.MonthEnd(test_months)
    train = frame.loc[:cutoff]
    test  = frame.loc[cutoff + pd.offsets.MonthEnd(0):]
    return train, test

train_df, test_df = chronosplit(XY, TEST_WINDOW_MONTHS)
X_train, y_train = train_df[features], train_df["target"]
X_test,  y_test  = test_df[features],  test_df["target"]

print("\nTrain/Test sizes:", X_train.shape, X_test.shape)

y_pred_naive = X_test["inf_mom_l1"].values
baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_naive))
baseline_mae  = mean_absolute_error(y_test, y_pred_naive)
print(f"\nBaseline  RMSE={baseline_rmse:.3f}  MAE={baseline_mae:.3f}")

# --- CV inside train only ---
tscv = TimeSeriesSplit(n_splits=5)

# OLS
ols_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ols", LinearRegression())
])
ols_pipe.fit(X_train, y_train)
pred_ols   = ols_pipe.predict(X_test)
ols_rmse   = np.sqrt(mean_squared_error(y_test, pred_ols))
ols_mae    = mean_absolute_error(y_test, pred_ols)
print(f"OLS       RMSE={ols_rmse:.3f}  MAE={ols_mae:.3f}")

ridge_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])
alphas = np.logspace(-6, 6, 25)
grid = GridSearchCV(
    ridge_pipe,
    param_grid={"ridge__alpha": alphas},
    scoring="neg_mean_squared_error",
    cv=tscv,
    n_jobs=-1
)
grid.fit(X_train, y_train)
pred_ridge = grid.best_estimator_.predict(X_test)
ridge_rmse = np.sqrt(mean_squared_error(y_test, pred_ridge))
ridge_mae  = mean_absolute_error(y_test, pred_ridge)
print(f"Ridge(a={grid.best_params_['ridge__alpha']:.2e})  RMSE={ridge_rmse:.3f}  MAE={ridge_mae:.3f}")


print("\nSummary")
print("-------")
print("Train period:", X_train.index.min().date(), "→", X_train.index.max().date())
print("Test  period:", X_test.index.min().date(),  "→", X_test.index.max().date())
print(f"Baseline RMSE={baseline_rmse:.3f} | OLS RMSE={ols_rmse:.3f} | Ridge RMSE={ridge_rmse:.3f}")


def rolling_backtest(X: pd.DataFrame,
                     y: pd.Series,
                     model,                    # e.g., Pipeline([("scaler",...), ("ridge", ...)])
                     start_date: pd.Timestamp,
                     window: str | None = None # None = expanding; or "36M" for sliding 36-month window
                    ) -> pd.DataFrame:
    """
    Rolling-origin backtest:
    - If window is None: expanding train window up to t-1.
    - If window = '36M' (example): use only the last 36 months up to t-1 (sliding window).
    Returns a DataFrame with columns: y, pred, naive, and the model's date index.
    """
    preds, naive, truth, dates = [], [], [], []

    # Ensure aligned and sorted
    X = X.sort_index()
    y = y.sort_index()
    assert X.index.equals(y.index), "X and y must share the same index"

    test_index = X.loc[start_date:].index
    for dt in test_index:
        # Training mask
        if window is None:
            train_mask = X.index < dt  # expanding
        else:
            # sliding window: keep observations in (dt - window, dt)
            left = dt - pd.tseries.frequencies.to_offset(window)
            train_mask = (X.index < dt) & (X.index > left)

        X_tr, y_tr = X.loc[train_mask], y.loc[train_mask]
        X_te = X.loc[[dt]]
        y_te = y.loc[dt]

        # Fit fresh each step (so no look-ahead)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)[0]

        # Naive baseline: next-month inflation ≈ last observed inflation (inf_mom_l1)
        naive_pred = X.loc[dt, "inf_mom_l1"]

        preds.append(pred)
        naive.append(naive_pred)
        truth.append(y_te)
        dates.append(dt)

    return pd.DataFrame({"y": truth, "pred": preds, "naive": naive}, index=dates)

bt_ridge = rolling_backtest(X, y, grid, start_date=pd.Timestamp("2015-01-31"))
bt_ols   = rolling_backtest(X, y, ols_pipe,   start_date=pd.Timestamp("2015-01-31"))

def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))
print("\nOLS backtest RMSE:", rmse(bt_ols["y"], bt_ols["pred"]))
print("Ridge backtest RMSE:", rmse(bt_ridge["y"], bt_ridge["pred"]))
