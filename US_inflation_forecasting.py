import os 
import pandas as pd
import numpy as np
from fredapi import Fred
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Defining functions

def get_fred() -> Fred:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("Missing FRED_API_KEY.")
    return Fred(api_key=api_key)


def to_month_end(s: pd.Series) -> pd.Series:
    s = s.copy(); s.index = pd.to_datetime(s.index)
    return s.resample("M").last()

def quarterly_to_monthly_ffill(s: pd.Series) -> pd.Series:
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    s.index = s.index.to_period("Q").to_timestamp("M", "end")
    s = s.asfreq("M").ffill()
    s.index = s.index.to_period("M").to_timestamp("M")
    return s

def inflation_mom(cpi: pd.Series) -> pd.Series:
    return 100 * (np.log(cpi) - np.log(cpi.shift(1)))

def add_lags(df: pd.DataFrame, cols, lags):
    df = df.copy()
    for c in cols:
        for L in lags:
            df[f"{c}_l{L}"] = df[c].shift(L)
    return df

def chronologicalsplit(frame: pd.DataFrame, test_months: int):
    cutoff = frame.index.max() - pd.offsets.MonthEnd(test_months)
    train = frame.loc[:cutoff]
    test  = frame.loc[cutoff + pd.offsets.MonthEnd(0):]
    return train, test

def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_holdout(XY, features, tscv):
    X, y = XY[features], XY["target"]
    train_df, test_df = chronologicalsplit(XY, TEST_WINDOW_MONTHS)
    X_train, y_train = train_df[features], train_df["target"]
    X_test,  y_test  = test_df[features],  test_df["target"]

    # Baseline (using last period's inflation)
    y_pred_naive = X_test["inf_mom_l1"].values
    baseline_rmse = (rmse(y_test, y_pred_naive))
    baseline_mae =(mean_absolute_error(y_test, y_pred_naive))

    # OLS
    ols = Pipeline([("scaler", StandardScaler()), ("ols", LinearRegression())])
    ols.fit(X_train, y_train)
    pred_ols = ols.predict(X_test)
    ols_rmse = (rmse(y_test, pred_ols)) 
    ols_mae = (mean_absolute_error(y_test, pred_ols))

    # Ridge 
    ridge_pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    alphas = np.logspace(-6, 6, 25)
    grid = GridSearchCV(ridge_pipe, {"ridge__alpha": alphas}, scoring="neg_mean_squared_error", cv=tscv, n_jobs=-1)
    grid.fit(X_train, y_train)
    pred_ridge = grid.best_estimator_.predict(X_test)
    ridge_rmse = (rmse(y_test, pred_ridge))
    ridge_mae = (mean_absolute_error(y_test, pred_ridge))

    return {"baseline_rmse": baseline_rmse, "baseline_mae":baseline_mae, 
            "ols_rmse": ols_rmse,"ols_mae": ols_mae, 
            "ridge_rmse": ridge_rmse, "ridge_mae":ridge_mae, 
            "best_alpha": grid.best_params_["ridge__alpha"]}

def rolling_backtest(X: pd.DataFrame,
                     y: pd.Series,
                     model,                    
                     start_date: pd.Timestamp,
                     window: str | None = None 
                    ) -> pd.DataFrame:
    preds, naive, truth, dates = [], [], [], []

    X = X.sort_index()
    y = y.sort_index()
    assert X.index.equals(y.index)

    test_index = X.loc[start_date:].index
    for dt in test_index:
        if window is None:
            train_mask = X.index < dt  
        else:
            left = dt - pd.tseries.frequencies.to_offset(window)
            train_mask = (X.index < dt) & (X.index > left)

        X_tr, y_tr = X.loc[train_mask], y.loc[train_mask]
        X_te = X.loc[[dt]]
        y_te = y.loc[dt]

        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)[0]

        naive_pred = X.loc[dt, "inf_mom_l1"]

        preds.append(pred)
        naive.append(naive_pred)
        truth.append(y_te)
        dates.append(dt)

    return pd.DataFrame({"y": truth, "pred": preds, "naive": naive}, index=dates)

# Retrieving data from FRED
fred = get_fred()

FRED_SERIES = {
    "CPI": "CPIAUCSL",
    "UNRATE": "UNRATE",
    "FEDFUNDS": "FEDFUNDS",
    "WTI": "MCOILWTICO",
    "RGDP": "GDPC1",  
}
# Forcing series to month-end
cpi     = to_month_end(fred.get_series(FRED_SERIES["CPI"]))
unrate  = to_month_end(fred.get_series(FRED_SERIES["UNRATE"]))
fedfunds= to_month_end(fred.get_series(FRED_SERIES["FEDFUNDS"]))
wti     = to_month_end(fred.get_series(FRED_SERIES["WTI"]))
rgdp_q  = fred.get_series(FRED_SERIES["RGDP"])
rgdp    = quarterly_to_monthly_ffill(rgdp_q)

# Aligning series to same start and end date 
series = [cpi, unrate, fedfunds, wti, rgdp]
start  = max(s.dropna().index.min() for s in series)
end    = min(s.dropna().index.max() for s in series)
aligned = [s.loc[start:end] for s in series]

df = pd.concat(aligned, axis=1)
df.columns = ["CPI", "UNRATE", "FEDFUNDS", "WTI", "RGDP"]

df["inf_mom"] = inflation_mom(df["CPI"])
df["target"]  = df["inf_mom"].shift(-1)  

# Creating lagged features
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

TEST_WINDOW_MONTHS = 36  # Setting the test window
tscv = TimeSeriesSplit(n_splits=5)

# Model with 1 lag, no dummies
results = evaluate_holdout(XY, lag_feature_cols, tscv)
print("\n\033[4mRMSE - holdout with 1 lag only (no seasonality dummies):\033[0m")
print(f"\nBaseline RMSE: {results['baseline_rmse']:.3f}")
print(f"OLS RMSE: {results['ols_rmse']:.3f}")
print(f"Ridge RMSE: {results['ridge_rmse']:.3f}")


df["inf_mom"] = inflation_mom(df["CPI"])
df["target"]  = df["inf_mom"].shift(-1)  

df["month"] = df.index.month
month_dummies = pd.get_dummies(df["month"], prefix="m", drop_first=True)  
df = pd.concat([df, month_dummies], axis=1)

df = add_lags(df, lag_cols, lags=[1, 3, 6, 12])
lag_feature_cols = [
    c for c in df.columns
    if any(c.endswith(suf) for suf in ["_l1","_l3","_l6","_l12"])
]

seasonality_cols = [c for c in df.columns if c.startswith("m_")]

features = lag_feature_cols + seasonality_cols

XY = df[features + ["target"]].dropna()
X = XY[features]
y = XY["target"]

# Model with lags 1, 3, 6 and 12 + seasonality dummies
tscv = TimeSeriesSplit(n_splits=5)
results = evaluate_holdout(XY, features, tscv)
print("\n\033[4mRMSE - holdout with lags 1, 3, 6, 12 and seasonality dummies:\033[0m")
print(f"\nBaseline RMSE: {results['baseline_rmse']:.3f}")
print(f"OLS RMSE: {results['ols_rmse']:.3f}")
print(f"Ridge RMSE: {results['ridge_rmse']:.3f}")

# Rolling Backtest scenario
best_alpha = results["best_alpha"]
ridge_best = Pipeline([
 ("scaler", StandardScaler()),
 ("ridge", Ridge(alpha=best_alpha))   
])

start_date = chronologicalsplit(XY, TEST_WINDOW_MONTHS)[1].index.min()

bt_ridge = rolling_backtest(X, y, ridge_best, start_date=start_date)
ridge_rmse_roll = (rmse(bt_ridge["y"], bt_ridge["pred"]))
naive_rmse_roll = (rmse(bt_ridge["y"], bt_ridge["naive"]))


ols_pipe = Pipeline([("scaler", StandardScaler()), ("ols", LinearRegression())])
bt_ols = rolling_backtest(X, y, ols_pipe, start_date=start_date) 
ols_rmse_roll = (rmse(bt_ols["y"],bt_ols["pred"]))
print("\n\n\033[4mRMSE - rolling backtest :\033[0m")
print(f"\nNaive Rolling RMSE  : {naive_rmse_roll:.3f}")
print(f"Rolling RMSE OLS : {ols_rmse_roll:.3f}")
print(f"Rolling RMSE Ridge : {ridge_rmse_roll:.3f}")