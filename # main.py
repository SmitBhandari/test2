# main.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima

# Create a simple dataset: Monthly sales data for 24 months (2 years)
np.random.seed(0)
months = pd.date_range('2022-01-01', periods=24, freq='M')
sales = np.random.normal(loc=200, scale=20, size=24).cumsum()  # Cumulative sales data

data = pd.DataFrame({'Month': months, 'Sales': sales})
data.set_index('Month', inplace=True)

# Display dataset in Streamlit
st.title('Sales Forecasting with auto_arima')
st.write('Here is the simple dataset:')
st.write(data)

# Fit the auto_arima model
model = auto_arima(data['Sales'], seasonal=True, m=12, trace=True, suppress_warnings=True)

# Forecast for the next 6 months
forecast = model.predict(n_periods=6)
forecast_index = pd.date_range(data.index[-1] + pd.Timedelta(days=30), periods=6, freq='M')

forecast_df = pd.DataFrame({'Month': forecast_index, 'Forecasted Sales': forecast})
forecast_df.set_index('Month', inplace=True)

# Plotting the sales data and forecast
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data.index, data['Sales'], label='Actual Sales', color='blue')
ax.plot(forecast_df.index, forecast_df['Forecasted Sales'], label='Forecasted Sales', color='red', linestyle='--')

ax.set_xlabel('Month')
ax.set_ylabel('Sales')
ax.set_title('Sales Forecasting using auto_arima')
ax.legend()

# Show the plot in Streamlit
st.pyplot(fig)

# Display forecasted values in Streamlit
st.write('Forecast for the next 6 months:')
st.write(forecast_df)
