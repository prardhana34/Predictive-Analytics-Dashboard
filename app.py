import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from datetime import timedelta

st.set_page_config(
    page_title="📊 Predictive Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGE_CSS = """
<style>
    .main { background-color: #0e1117; color: #f2f2f2; }
    .stApp { background-color: #0e1117; color: #f2f2f2; }
    .css-1l02zno { background-color: #111827; }
    .css-1dq8tca { background-color: #111827; }
    .css-1q8dd3e { background-color: #111827; }
    .css-1v0mbdj { background-color: #111827; }
    .stSidebar { background-color: #111827; }
    .reportview-container .main .block-container { padding-top: 1rem; padding-right: 1rem; padding-left: 1rem; }
    .metric-value { color: #7dd3fc; }
    .metric-label { color: #d1d5db; }
    .stButton>button { background-color: #2563eb; color: white; }
    .stDownloadButton>button { background-color: #2563eb; color: white; }
</style>
"""

REQUIRED_COLUMNS = [
    "Date",
    "Sales",
    "Revenue",
    "Profit",
    "Customers",
    "Product Category",
    "Region",
]


@st.cache_data(show_spinner=False)
def generate_sample_data(rows=1096):
    np.random.seed(42)
    start_date = pd.Timestamp("2020-01-01")
    dates = pd.date_range(start_date, periods=rows, freq="D")

    categories = ["Electronics", "Office Supplies", "Apparel", "Home & Garden"]
    regions = ["North America", "EMEA", "APAC", "Latin America"]

    trend = np.linspace(250, 850, rows)
    seasonal = 80 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    noise = np.random.normal(0, 40, rows)
    sales = np.clip(np.round(trend + seasonal + noise), 40, None).astype(int)

    price = np.random.uniform(45, 110, rows)
    revenue = np.round(sales * price * np.random.uniform(0.95, 1.08, rows), 2)
    profit_margin = np.clip(np.random.normal(0.18, 0.03, rows), 0.1, 0.28)
    profit = np.round(revenue * profit_margin, 2)
    customers = np.clip(np.round(sales * np.random.uniform(0.55, 0.85, rows) + np.random.normal(0, 20, rows)), 10, None).astype(int)

    data = pd.DataFrame(
        {
            "Date": dates,
            "Sales": sales,
            "Revenue": revenue,
            "Profit": profit,
            "Customers": customers,
            "Product Category": np.random.choice(categories, rows, p=[0.3, 0.25, 0.2, 0.25]),
            "Region": np.random.choice(regions, rows, p=[0.35, 0.25, 0.2, 0.2]),
        }
    )

    missing_revenue = np.random.choice(rows, size=12, replace=False)
    data.loc[missing_revenue, "Revenue"] = np.nan
    missing_customers = np.random.choice(rows, size=10, replace=False)
    data.loc[missing_customers, "Customers"] = np.nan
    duplicate_rows = data.sample(3, random_state=24)
    data = pd.concat([data, duplicate_rows], ignore_index=True).sample(frac=1, random_state=13).reset_index(drop=True)
    data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")
    return data


@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        sample_path = Path(__file__).parent / "sample_historical_data.csv"
        if sample_path.exists():
            df = pd.read_csv(sample_path)
        else:
            df = generate_sample_data()
    return df


@st.cache_data(show_spinner=False)
def preprocess_data(df, normalize=False):
    data = df.copy()

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    data = data.drop_duplicates()
    data = data.dropna(subset=["Date"])

    numeric_columns = ["Sales", "Revenue", "Profit", "Customers"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            data[col] = data[col].fillna(data[col].median())

    if "Product Category" in data.columns:
        data["Product Category"] = data["Product Category"].fillna("Unknown")
    if "Region" in data.columns:
        data["Region"] = data["Region"].fillna("Unknown")

    data = data.sort_values("Date").reset_index(drop=True)
    data["Month"] = data["Date"].dt.month
    data["Quarter"] = data["Date"].dt.quarter
    data["DayOfWeek"] = data["Date"].dt.dayofweek
    data["DayOfMonth"] = data["Date"].dt.day
    data["IsWeekend"] = data["DayOfWeek"].isin([5, 6]).astype(int)
    data["Sales_Rolling_7"] = data["Sales"].rolling(7, min_periods=1).mean()
    data["Revenue_Rolling_7"] = data["Revenue"].rolling(7, min_periods=1).mean()
    data["Profit_Rolling_7"] = data["Profit"].rolling(7, min_periods=1).mean()
    data["Customers_Rolling_7"] = data["Customers"].rolling(7, min_periods=1).mean()
    data["DateOrdinal"] = data["Date"].map(pd.Timestamp.toordinal)

    normalized = data.copy()
    if normalize:
        scale_cols = [col for col in numeric_columns if col in normalized.columns]
        normalized[scale_cols] = (normalized[scale_cols] - normalized[scale_cols].min()) / (
            normalized[scale_cols].max() - normalized[scale_cols].min()
        )

    return data, normalized


def create_features(df):
    feature_columns = ["DateOrdinal", "Month", "DayOfWeek", "Quarter", "DayOfMonth", "IsWeekend"]
    return df[feature_columns]


def compute_metrics(true_values, predicted_values):
    return {
        "R2": r2_score(true_values, predicted_values),
        "MAE": mean_absolute_error(true_values, predicted_values),
        "RMSE": np.sqrt(mean_squared_error(true_values, predicted_values)),
    }


def train_models(df, target):
    train_index = int(len(df) * 0.8)
    train_df = df.iloc[:train_index]
    test_df = df.iloc[train_index:]

    X_train = create_features(train_df)
    y_train = train_df[target]
    X_test = create_features(test_df)
    y_test = test_df[target]

    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    return {
        "train_df": train_df,
        "test_df": test_df,
        "lr": lr,
        "rf": rf,
        "metrics": {
            "Linear Regression": compute_metrics(y_test, y_pred_lr),
            "Random Forest": compute_metrics(y_test, y_pred_rf),
        },
        "predictions": {
            "linear": y_pred_lr,
            "forest": y_pred_rf,
        },
    }


def forecast_future(df, model, days, target, method_name="Forecast"):
    last_date = df["Date"].max()
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=days, freq="D")
    future = pd.DataFrame({"Date": future_dates})
    future["Month"] = future["Date"].dt.month
    future["Quarter"] = future["Date"].dt.quarter
    future["DayOfWeek"] = future["Date"].dt.dayofweek
    future["DayOfMonth"] = future["Date"].dt.day
    future["IsWeekend"] = future["DayOfWeek"].isin([5, 6]).astype(int)
    future["DateOrdinal"] = future["Date"].map(pd.Timestamp.toordinal)
    future[target] = model.predict(create_features(future))
    future[target] = future[target].clip(lower=0)
    future["Model"] = method_name
    return future


def forecast_time_series(df, target, days):
    trend_model = LinearRegression()
    X = df[["DateOrdinal"]]
    y = df[target]
    trend_model.fit(X, y)
    last_date = df["Date"].max()
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=days, freq="D")
    future = pd.DataFrame({"Date": future_dates})
    future["Month"] = future["Date"].dt.month
    future["DateOrdinal"] = future["Date"].map(pd.Timestamp.toordinal)

    trend_forecast = trend_model.predict(future[["DateOrdinal"]])
    df = df.copy()
    df["Residual"] = y - trend_model.predict(X)
    seasonal_map = df.groupby("Month")["Residual"].mean().to_dict()
    future["Seasonal"] = future["Month"].map(lambda x: seasonal_map.get(x, 0))
    future[target] = (trend_forecast + future["Seasonal"]).clip(lower=0)
    future["Model"] = "Time Series"
    return future


def format_metrics(metrics):
    return {
        "R2": f"{metrics['R2']:.2f}",
        "MAE": f"{metrics['MAE']:.2f}",
        "RMSE": f"{metrics['RMSE']:.2f}",
    }


def get_download_link(df, filename):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()


def main():
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    st.title("📊 Predictive Analytics Dashboard")
    st.markdown(
        "Use historical sales and revenue data to uncover trends, validate models, and forecast future performance with regression and time-series forecasting."
    )

    uploaded_file = st.sidebar.file_uploader("📥 Upload historical dataset", type=["csv"])
    forecast_days = st.sidebar.selectbox("🔮 Forecast period", [30, 60, 90], index=0)
    normalize = st.sidebar.checkbox("⚙️ Normalize numeric values", value=False)
    show_data = st.sidebar.checkbox("📄 Show raw dataset", value=False)

    raw_data = load_data(uploaded_file)

    if not all(col in raw_data.columns for col in REQUIRED_COLUMNS):
        st.error(
            "The uploaded dataset must contain the following columns: "
            + ", ".join(REQUIRED_COLUMNS)
        )
        st.stop()

    full_data, normalized_data = preprocess_data(raw_data, normalize=normalize)

    categories = sorted(full_data["Product Category"].unique())
    regions = sorted(full_data["Region"].unique())

    category_filter = st.sidebar.multiselect("Product categories", categories, default=categories)
    region_filter = st.sidebar.multiselect("Regions", regions, default=regions)

    min_date = full_data["Date"].min()
    max_date = full_data["Date"].max()
    selected_date_range = st.sidebar.date_input(
        "Date range", [min_date, max_date], min_value=min_date, max_value=max_date
    )

    filtered_data = full_data[
        (full_data["Product Category"].isin(category_filter))
        & (full_data["Region"].isin(region_filter))
        & (full_data["Date"] >= pd.to_datetime(selected_date_range[0]))
        & (full_data["Date"] <= pd.to_datetime(selected_date_range[1]))
    ].copy()

    if filtered_data.empty:
        st.warning("No data matches the selected filters. Please adjust the date range, categories, or regions.")
        st.stop()

    if show_data:
        st.subheader("Raw business dataset")
        st.dataframe(filtered_data.head(20))

    kpi_sales = int(filtered_data["Sales"].sum())
    kpi_revenue = float(filtered_data["Revenue"].sum())
    kpi_profit = float(filtered_data["Profit"].sum())
    kpi_customers = int(filtered_data["Customers"].sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", f"{kpi_sales:,}", delta=f"{int(filtered_data['Sales'].diff().mean()):,} avg/day")
    col2.metric("Total Revenue", f"${kpi_revenue:,.0f}", delta=f"${filtered_data['Revenue'].diff().mean():.0f} avg/day")
    col3.metric("Total Profit", f"${kpi_profit:,.0f}", delta=f"${filtered_data['Profit'].diff().mean():.0f} avg/day")
    col4.metric("Total Customers", f"{kpi_customers:,}", delta=f"{int(filtered_data['Customers'].diff().mean()):,} avg/day")

    with st.expander("🧹 Dataset summary and preprocessing details"):
        st.write(
            "After loading and cleaning the data, the dashboard removes duplicates, fills missing numeric values with median values, and adds date-based features for better forecasting."
        )
        st.write(filtered_data.describe(include="all"))
        st.write("Sample of data used for modeling:")
        st.dataframe(filtered_data.head(10))

    model_target = st.sidebar.selectbox(
        "Primary forecast metric", ["Sales", "Revenue", "Profit", "Customers"], index=0
    )

    st.markdown("---")

    model_results = train_models(filtered_data, model_target)

    forecast_lr = forecast_future(filtered_data, model_results["lr"], forecast_days, model_target, "Linear Regression")
    forecast_rf = forecast_future(filtered_data, model_results["rf"], forecast_days, model_target, "Random Forest")
    forecast_ts = forecast_time_series(filtered_data, model_target, forecast_days)

    forecast_comparison = pd.concat([forecast_lr, forecast_rf, forecast_ts], ignore_index=True)

    last_actual = filtered_data[model_target].iloc[-30:].mean()
    forecast_growth_pct = ((forecast_comparison[model_target].mean() / last_actual) - 1) * 100
    forecast_direction = "growth" if forecast_growth_pct >= 0 else "decline"

    st.header("🔮 Forecast dashboard")

    st.info(
        f"The selected forecast metric is {model_target}. The average {model_target.lower()} over the next {forecast_days} days is projected to {forecast_direction} by {forecast_growth_pct:.1f}% compared to the last 30-day historical average. "
    )

    history_fig = px.line(
        filtered_data,
        x="Date",
        y=["Sales", "Revenue", "Profit", "Customers"],
        title="Historical business performance",
        labels={"value": "Amount", "variable": "Indicator"},
        template="plotly_dark",
    )
    history_fig.update_layout(legend=dict(title="Metric", orientation="h", y=1.05, x=0.1))

    forecast_fig = px.line(
        forecast_comparison,
        x="Date",
        y=model_target,
        color="Model",
        title=f"{model_target} forecast comparison",
        template="plotly_dark",
    )
    forecast_fig.add_scatter(
        x=filtered_data["Date"],
        y=filtered_data[model_target],
        mode="lines",
        name="History",
        line=dict(color="#ffffff", width=2, dash="dash"),
    )

    col1, col2 = st.columns((2, 1))
    col1.plotly_chart(history_fig, use_container_width=True)
    col2.plotly_chart(forecast_fig, use_container_width=True)

    st.markdown("---")

    st.subheader("📊 Model performance and comparison")
    metrics_display = pd.DataFrame(
        {
            method: format_metrics(vals)
            for method, vals in model_results["metrics"].items()
        }
    ).rename(index={"R2": "R²", "MAE": "Mean Absolute Error", "RMSE": "Root Mean Square Error"})
    st.table(metrics_display)

    actual_vs_pred = filtered_data.iloc[len(filtered_data) - len(model_results["test_df"]):].copy()
    actual_vs_pred["Linear Regression"] = model_results["predictions"]["linear"]
    actual_vs_pred["Random Forest"] = model_results["predictions"]["forest"]

    scatter_fig = px.scatter(
        actual_vs_pred,
        x=model_target,
        y="Linear Regression",
        trendline="ols",
        labels={model_target: "Actual", "Linear Regression": "Predicted"},
        title=f"Actual vs Predicted {model_target} (Linear Regression)",
        template="plotly_dark",
    )
    scatter_fig.add_scatter(
        x=actual_vs_pred[model_target],
        y=actual_vs_pred["Random Forest"],
        mode="markers",
        name="Random Forest",
        marker=dict(color="#f97316"),
    )

    st.plotly_chart(scatter_fig, use_container_width=True)

    st.markdown("---")

    st.subheader("🌐 Category and region performance")
    col1, col2 = st.columns(2)
    category_fig = px.bar(
        filtered_data.groupby("Product Category")[["Sales", "Revenue", "Profit"]]
        .sum()
        .reset_index()
        .melt(id_vars=["Product Category"], var_name="Metric", value_name="Value"),
        x="Product Category",
        y="Value",
        color="Metric",
        barmode="group",
        title="Revenue, Sales and Profit by Category",
        template="plotly_dark",
    )
    region_fig = px.bar(
        filtered_data.groupby("Region")[["Sales", "Revenue", "Profit"]]
        .sum()
        .reset_index()
        .melt(id_vars=["Region"], var_name="Metric", value_name="Value"),
        x="Region",
        y="Value",
        color="Metric",
        barmode="group",
        title="Business performance by region",
        template="plotly_dark",
    )
    col1.plotly_chart(category_fig, use_container_width=True)
    col2.plotly_chart(region_fig, use_container_width=True)

    st.markdown("---")

    st.subheader("📈 Distribution, correlation, and forecast insights")
    col1, col2, col3 = st.columns((1.5, 1, 1))
    hist_fig = px.histogram(
        filtered_data,
        x="Revenue",
        nbins=35,
        title="Revenue distribution",
        template="plotly_dark",
    )
    heatmap_data = filtered_data[["Sales", "Revenue", "Profit", "Customers"]].corr()
    heatmap_fig = px.imshow(
        heatmap_data,
        text_auto=True,
        color_continuous_scale="Viridis",
        title="Correlation heatmap",
        template="plotly_dark",
    )
    trend_fig = px.line(
        filtered_data,
        x="Date",
        y=["Sales_Rolling_7", "Revenue_Rolling_7", "Profit_Rolling_7", "Customers_Rolling_7"],
        title="7-day rolling trend indicators",
        labels={"value": "Value", "variable": "Rolling Metric"},
        template="plotly_dark",
    )
    col1.plotly_chart(hist_fig, use_container_width=True)
    col2.plotly_chart(heatmap_fig, use_container_width=True)
    col3.plotly_chart(trend_fig, use_container_width=True)

    st.markdown("---")

    st.subheader("⬇️ Forecast download")
    forecast_file = get_download_link(forecast_comparison, f"forecast_{model_target.lower()}_{forecast_days}d.csv")
    st.download_button(
        label="Download forecast data",
        data=forecast_file,
        file_name=f"forecast_{model_target.lower()}_{forecast_days}d.csv",
        mime="text/csv",
    )

    st.markdown("---")

    st.subheader("💡 Business insights")
    st.write(
        "- Strong sale momentum is visible in the latest period, with revenue and customer growth aligned to the increasing trend."
    )
    st.write(
        "- The forecast comparison shows how linear regression and random forest models behave similarly in the short-term, while the time-series forecast highlights seasonal cycles."
    )
    st.write(
        f"- Use the dashboard filters to compare product categories and regions, and observe how different segments influence profit and revenue growth over time."
    )
    st.write(
        f"- For the next {forecast_days} days, the model suggests a {forecast_direction} trend in {model_target.lower()}. Align inventory, promotions, and customer outreach to capture this projected movement."
    )

    st.write("---")
    st.write("Built with Streamlit, Plotly, scikit-learn, pandas, and NumPy for a complete predictive analytics experience.")


if __name__ == "__main__":
    main()
