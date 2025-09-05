from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("ev_load_forecast_and_recommendation.pkl")

def prepare_future_features(last_timestamp, hours=24, static_features=None):
    future_hours = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=hours, freq='H')
    df = pd.DataFrame({'Timestamp': future_hours})

    df['Day_of_Week'] = df['Timestamp'].dt.dayofweek
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
    df['IsWeekend'] = df['Day_of_Week'].isin([5,6]).astype(int)

    defaults = {
        'Fleet_Size': 100,
        'Average_Battery_Capacity_kWh': 40,
        'Number_of_Charging_Stations': 10,
        'Charging_Power_Rating_kW': 7.4,
        'Charging_Efficiency': 0.9,
        'Total_Distance_Driven_km': 50,
        'Average_Speed_kmh': 40,
        'Loading_Unloading_Times_hours': 1,
        'Temperature_C': 25,
        'Humidity_%': 60,
        'Previous_Charging_Loads_kW': 5,
        'Charging_Duration_hours': 1,
        'Electricity_Prices_USD': 0.12,
        'Grid_Demand_MW': 500,
        'Load_per_EV': 5,
        'Power_per_Station': 3
    }
    if static_features:
        defaults.update(static_features)

    for feat, val in defaults.items():
        df[feat] = val

    feature_cols = [
        'Day_of_Week', 'Fleet_Size', 'Average_Battery_Capacity_kWh',
        'Number_of_Charging_Stations', 'Charging_Power_Rating_kW',
        'Charging_Efficiency', 'Total_Distance_Driven_km', 'Average_Speed_kmh',
        'Loading_Unloading_Times_hours', 'Temperature_C', 'Humidity_%',
        'Previous_Charging_Loads_kW', 'Charging_Duration_hours',
        'Electricity_Prices_USD', 'Grid_Demand_MW', 'Hour', 'DayOfWeek',
        'Month', 'IsWeekend', 'Load_per_EV', 'Power_per_Station'
    ]
    return df[feature_cols + ['Timestamp']]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get inputs from form
    last_timestamp = pd.Timestamp(request.form['last_timestamp'])
    hours = int(request.form.get('hours', 24))
    start_hour = int(request.form['start_hour'])
    end_hour = int(request.form['end_hour'])

    # Prepare future data
    future_df = prepare_future_features(last_timestamp, hours=hours)
    X_future = future_df.drop(columns=['Timestamp'])

    # Predict load
    future_df['Predicted_Load_kW'] = model.predict(X_future)

    # Filter for best times
    filtered = future_df[(future_df['Hour'] >= start_hour) & (future_df['Hour'] <= end_hour)]
    recommended = filtered.nsmallest(3, 'Predicted_Load_kW')

    # Prepare output
    results = {
        "all_predictions": future_df[['Timestamp', 'Hour', 'Predicted_Load_kW']].to_dict(orient='records'),
        "best_times": recommended[['Timestamp', 'Predicted_Load_kW']].to_dict(orient='records')
    }

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
