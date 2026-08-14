import pandas as pd

# Load wildfire weather data.
data_path = "./data/wildfire/input/"
df = pd.read_csv(data_path + "Fire_Weather_Data_2002-2014_2016.csv")

# Convert relevant columns to numeric values; invalid entries become NaN.
df["avrh_mean"] = pd.to_numeric(df["avrh_mean"], errors="coerce")
df["fatalities_last"] = pd.to_numeric(df["fatalities_last"], errors="coerce")

# Compute the overall mean number of fatalities.
overall_mean = df["fatalities_last"].mean()

# Compute the mean fatalities for records with humidity strictly below 30%.
low_humidity_mean = df[df["avrh_mean"] < 30]["fatalities_last"].mean()

# Return the difference between the low-humidity mean and the overall mean.
answer = low_humidity_mean - overall_mean
print(round(answer, 4))