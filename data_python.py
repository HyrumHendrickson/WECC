import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '2000-2020-rexburg.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Display dataset info
print("\nDataset Information:")
data.info()

# Display basic statistics of numerical columns
print("\nSummary statistics:")
print(data.describe())

# Line plot for temperature over time
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Temperature (F)'], label='Temperature', color='orange')
plt.title('Temperature Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°F)')
plt.legend()
plt.show()



