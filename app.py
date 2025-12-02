import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

st.title("Solar Power Generation Prediction")
st.write("XGBoost model")

@st.cache_data
def load_data():
    df = pd.read_csv("solarpowergeneration (1).csv")
    df['average-wind-speed-(period)'].fillna(
        df['average-wind-speed-(period)'].mean(),
        inplace=True
    )

    return df

df = load_data()

st.subheader("Dataset Preview")
st.write(df.head())


if "visibility" in df.columns:
    df = df.drop(columns=["visibility"])


X = df.drop(columns=["power-generated"])
y = df["power-generated"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


@st.cache_resource
def train_xgb():
    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    xgb.fit(X_train_scaled, y_train)
    xgb_params = {
        "n_estimators": [200, 300, 400],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 6, 8],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0]
    }

    grid = GridSearchCV(
        XGBRegressor(random_state=42, objective="reg:squarederror"),
        param_grid=xgb_params,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(X_train_scaled, y_train)

    return grid.best_estimator_

best_xgb = train_xgb()


y_pred = best_xgb.predict(X_test_scaled)

st.subheader("Model Performance (Tuned XGBoost)")
st.write("*MAE:*", round(mean_absolute_error(y_test, y_pred), 3))
st.write("*RMSE:*", round(np.sqrt(mean_squared_error(y_test, y_pred)), 3))
st.write("*R2 Score:*", round(r2_score(y_test, y_pred), 4))


st.subheader("Enter Values to Predict Power Generation")

def get_user_inputs():
    input_data = {}
    for col in X.columns:
        default_val = float(df[col].mean())
        input_data[col] = st.number_input(f"Enter {col}", value=default_val)
    return pd.DataFrame([input_data])

input_df = get_user_inputs()

if st.button("Predict Power Generated"):
    scaled_input = scaler.transform(input_df)
    prediction = best_xgb.predict(scaled_input)[0]

    st.success(f"Predicted Power Generated: *{prediction:.2f} units*")
