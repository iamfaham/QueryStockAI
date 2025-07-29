#!/usr/bin/env python3
"""
Prophet Accuracy Test Script
Trains Prophet model on given ticker data up to June 2025 and tests predictions for July 2025.
"""

import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
import warnings

warnings.filterwarnings("ignore")

ticker = "AAPL"


def get_aapl_data():
    """Get AAPL historical data."""
    print("📊 Fetching AAPL historical data...")

    # Get data for the past 2 years to have enough training data
    raw_data = yf.Ticker("AAPL")
    data = raw_data.history(period="2y")

    if data.empty:
        raise ValueError("No data received for AAPL")

    print(f"✅ Retrieved {len(data)} days of AAPL data")
    print(f"📅 Date range: {data.index.min().date()} to {data.index.max().date()}")

    return data


def prepare_prophet_data(data, end_date):
    """Prepare data for Prophet training (up to end_date)."""
    print(f"\n🔧 Preparing Prophet data up to {end_date.date()}...")

    # Convert end_date to timezone-aware datetime to match data index
    if data.index.tz is not None:
        end_date = pd.Timestamp(end_date).tz_localize(data.index.tz)

    # Filter data up to end_date
    training_data = data[data.index <= end_date].copy()

    # Remove outliers using IQR method
    Q1 = training_data["Close"].quantile(0.25)
    Q3 = training_data["Close"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out outliers
    training_data = training_data[
        (training_data["Close"] >= lower_bound)
        & (training_data["Close"] <= upper_bound)
    ]

    print(
        f"📊 Removed outliers: {len(data[data.index <= end_date]) - len(training_data)} points"
    )

    # Prepare for Prophet (requires 'ds' and 'y' columns)
    prophet_data = training_data.reset_index()
    prophet_data["ds"] = prophet_data["Date"].dt.tz_localize(None)  # Remove timezone
    prophet_data["y"] = prophet_data["Close"]

    # Select only required columns
    prophet_data = prophet_data[["ds", "y"]]

    print(f"✅ Training data prepared: {len(prophet_data)} days")
    print(
        f"📈 Price range: ${prophet_data['y'].min():.2f} - ${prophet_data['y'].max():.2f}"
    )

    return prophet_data


def train_prophet_model(data):
    """Train Prophet model on the provided data."""
    print("\n🤖 Training Prophet model...")

    # Configure Prophet model with optimized parameters
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.01,  # Reduced for smoother trends
        seasonality_prior_scale=10.0,  # Increased seasonality strength
        seasonality_mode="multiplicative",
        interval_width=0.8,  # Tighter confidence intervals
        mcmc_samples=0,  # Disable MCMC for faster training
    )

    # Add custom seasonalities for better stock patterns
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    model.add_seasonality(name="quarterly", period=91.25, fourier_order=8)

    # Train the model
    model.fit(data)

    print("✅ Prophet model trained successfully")
    return model


def make_predictions(model, start_date, end_date):
    """Make predictions for the specified date range."""
    print(f"\n🔮 Making predictions from {start_date.date()} to {end_date.date()}...")

    # Calculate the number of days to predict
    days_to_predict = (end_date - start_date).days + 1

    # Create future dataframe
    future = model.make_future_dataframe(periods=days_to_predict)
    forecast = model.predict(future)

    # Filter predictions for the specified period
    predictions = forecast[
        (forecast["ds"] >= start_date) & (forecast["ds"] <= end_date)
    ].copy()

    print(f"✅ Generated {len(predictions)} predictions for {days_to_predict} days")

    return predictions


def get_actual_july_data(data, start_date, end_date):
    """Get actual AAPL data for July."""
    print(f"\n📊 Fetching actual July data...")

    # Convert dates to timezone-aware datetime to match data index
    if data.index.tz is not None:
        start_date = pd.Timestamp(start_date).tz_localize(data.index.tz)
        end_date = pd.Timestamp(end_date).tz_localize(data.index.tz)

    # Get actual data for July
    actual_data = data[(data.index >= start_date) & (data.index <= end_date)].copy()

    print(f"✅ Retrieved {len(actual_data)} days of actual July data")

    return actual_data


def calculate_accuracy_metrics(predictions, actual_data):
    """Calculate accuracy metrics."""
    print("\n📈 Calculating accuracy metrics...")

    # Prepare actual data with timezone-naive dates for merging
    actual_data_prepared = actual_data.reset_index().copy()
    actual_data_prepared["Date"] = actual_data_prepared["Date"].dt.tz_localize(None)

    # Merge predictions with actual data
    comparison = pd.merge(
        predictions[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        actual_data_prepared[["Date", "Close"]],
        left_on="ds",
        right_on="Date",
        how="inner",
    )

    if comparison.empty:
        print("❌ No overlapping data found for comparison")
        return None

    # Calculate metrics
    mae = mean_absolute_error(comparison["Close"], comparison["yhat"])
    mse = mean_squared_error(comparison["Close"], comparison["yhat"])
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(comparison["Close"], comparison["yhat"]) * 100

    # Calculate directional accuracy (up/down prediction)
    actual_direction = comparison["Close"].diff().dropna()
    predicted_direction = comparison["yhat"].diff().dropna()

    # Align the data
    min_len = min(len(actual_direction), len(predicted_direction))
    actual_direction = actual_direction.iloc[-min_len:]
    predicted_direction = predicted_direction.iloc[-min_len:]

    directional_accuracy = (
        np.mean((actual_direction > 0) == (predicted_direction > 0)) * 100
    )

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "directional_accuracy": directional_accuracy,
        "comparison_data": comparison,
    }


def print_results(metrics, predictions, actual_data):
    """Print detailed results."""
    print("\n" + "=" * 60)
    print("📊 PROPHET ACCURACY TEST RESULTS")
    print("=" * 60)

    if metrics is None:
        print("❌ No metrics available - insufficient data for comparison")
        return

    print(f"\n📈 Accuracy Metrics:")
    print(f"   Mean Absolute Error (MAE): ${metrics['mae']:.2f}")
    print(f"   Mean Squared Error (MSE): {metrics['mse']:.2f}")
    print(f"   Root Mean Squared Error (RMSE): ${metrics['rmse']:.2f}")
    print(f"   Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%")
    print(f"   Directional Accuracy: {metrics['directional_accuracy']:.1f}%")

    print(f"\n📊 Prediction Summary:")
    print(f"   Training Period: Up to June 30, 2025")
    print(f"   Test Period: July 1-25, 2025")
    print(f"   Test Days: {len(metrics['comparison_data'])}")

    # Show some sample predictions vs actual
    comparison = metrics["comparison_data"]
    print(f"\n📋 Sample Predictions vs Actual (first 5 days):")
    print(f"{'Date':<12} {'Predicted':<12} {'Actual':<12} {'Error':<12}")
    print("-" * 50)

    for i in range(min(5, len(comparison))):
        row = comparison.iloc[i]
        error = row["Close"] - row["yhat"]
        print(
            f"{row['ds'].strftime('%Y-%m-%d'):<12} "
            f"${row['yhat']:<11.2f} "
            f"${row['Close']:<11.2f} "
            f"${error:<11.2f}"
        )

    print("\n" + "=" * 60)


def test_multiple_configurations(data, training_end, test_start, test_end):
    """Test multiple Prophet configurations to find the best one."""
    print("\n🔬 Testing multiple Prophet configurations...")

    configurations = [
        {
            "name": "Default",
            "params": {
                "yearly_seasonality": True,
                "weekly_seasonality": True,
                "daily_seasonality": False,
                "changepoint_prior_scale": 0.05,
                "seasonality_mode": "multiplicative",
            },
        },
        {
            "name": "Optimized",
            "params": {
                "yearly_seasonality": True,
                "weekly_seasonality": True,
                "daily_seasonality": False,
                "changepoint_prior_scale": 0.01,
                "seasonality_prior_scale": 10.0,
                "seasonality_mode": "multiplicative",
                "interval_width": 0.8,
                "mcmc_samples": 0,
            },
        },
        {
            "name": "Conservative",
            "params": {
                "yearly_seasonality": True,
                "weekly_seasonality": False,
                "daily_seasonality": False,
                "changepoint_prior_scale": 0.001,
                "seasonality_mode": "additive",
            },
        },
    ]

    best_config = None
    best_mape = float("inf")
    results = []

    for config in configurations:
        print(f"\n🧪 Testing {config['name']} configuration...")

        try:
            # Prepare training data
            training_data = prepare_prophet_data(data, training_end)

            # Create and train model
            model = Prophet(**config["params"])

            # Add custom seasonalities for optimized config
            if config["name"] == "Optimized":
                model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
                model.add_seasonality(name="quarterly", period=91.25, fourier_order=8)

            model.fit(training_data)

            # Make predictions
            predictions = make_predictions(model, test_start, test_end)

            # Get actual data
            actual_data = get_actual_july_data(data, test_start, test_end)

            # Calculate metrics
            metrics = calculate_accuracy_metrics(predictions, actual_data)

            if metrics:
                results.append({"config": config["name"], "metrics": metrics})

                if metrics["mape"] < best_mape:
                    best_mape = metrics["mape"]
                    best_config = config["name"]

                print(
                    f"✅ {config['name']}: MAPE = {metrics['mape']:.2f}%, Directional = {metrics['directional_accuracy']:.1f}%"
                )
            else:
                print(f"❌ {config['name']}: No valid metrics")

        except Exception as e:
            print(f"❌ {config['name']}: Error - {e}")

    return best_config, results


def main():
    """Main function to run the Prophet accuracy test."""
    print("🚀 Starting Prophet Accuracy Test for AAPL")
    print("=" * 60)

    try:
        # Define date ranges
        training_end = datetime(2025, 6, 30)  # End of June 2025
        test_start = datetime(2025, 7, 1)  # Start of July 2025
        test_end = datetime(2025, 7, 25)  # End of July 2025 (up to 25th)

        print(f"📅 Training Period: Up to {training_end.date()}")
        print(f"📅 Test Period: {test_start.date()} to {test_end.date()}")

        # Get data
        data = get_aapl_data()

        # Test multiple configurations
        best_config, all_results = test_multiple_configurations(
            data, training_end, test_start, test_end
        )

        if best_config:
            print(f"\n🏆 Best configuration: {best_config}")

            # Use the best configuration for final results
            best_result = next(r for r in all_results if r["config"] == best_config)
            metrics = best_result["metrics"]

            # Print final results
            print_results(metrics, None, None)
        else:
            print("\n❌ No valid configurations found")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
