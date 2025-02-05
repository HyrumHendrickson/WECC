import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
file_path = '../scraper/2000-2020-rexburg.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Convert 'Date' to datetime
data['Date'] = pd.to_datetime(data['Date'])
# Sort data by date
data = data.sort_values(by='Date')

# Seasonal decomposition
decomposed = seasonal_decompose(data['Temperature (F)'].dropna(), model='additive', period=365)
data['Trend'] = decomposed.trend
data['Seasonal'] = decomposed.seasonal
data['Residual'] = decomposed.resid

def plot_temperature():
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Temperature (F)'], label='Temperature', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°F)')
    plt.title('Temperature vs Time')
    plt.legend()
    plt.show()

def plot_decomposition():
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(data['Date'], data['Trend'], label='Trend', color='blue')
    axes[0].set_title('Trend Component')
    
    axes[1].plot(data['Date'], data['Seasonal'], label='Seasonality', color='green')
    axes[1].set_title('Seasonality Component')
    
    axes[2].plot(data['Date'], data['Residual'], label='Residual', color='gray')
    axes[2].set_title('Residual Component')
    
    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()

def plot_temp_without_seasonality():
    temp_without_seasonality = data['Temperature (F)'] - data['Seasonal']
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], temp_without_seasonality, label='Temperature (No Seasonality)', color='red')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°F)')
    plt.title('Temperature over Time Without Seasonality')
    plt.legend()
    plt.show()

# Execute plots
plot_temperature()
plot_decomposition()
plot_temp_without_seasonality()
