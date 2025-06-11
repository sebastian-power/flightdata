import requests
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from numpy.polynomial import Polynomial

url = "https://api.aviationstack.com/v1/timetable"
params = {
    "access_key": os.getenv("AVIATIONSTACK_API_KEY"),
    "iataCode": "SYD",
    "type": "departure",
}

response = requests.get("https://api.aviationstack.com/v1/timetable", params=params)
response.raise_for_status()
response = response.json()

# response = json.load(open("response.json"))


df = pd.json_normalize(response["data"])

arrival = df.groupby("arrival.iataCode").size().reset_index(name="count").sort_values(by="count", ascending=False)
airlines = df.groupby("airline.icaoCode").size().reset_index(name="count").sort_values(by="count", ascending=False)
airlines = airlines[airlines["count"] > 1]
arrival = arrival[arrival["count"] > 1]

departure_times = pd.DataFrame({"date": pd.to_datetime(df["departure.scheduledTime"])})
departure_times["hour"] = departure_times["date"].dt.floor('h')
departure_numbers = departure_times.groupby("hour").size()

full_range = pd.date_range(start=departure_numbers.index.min(), end=departure_numbers.index.max(), freq='h')
full_counts = departure_numbers.reindex(full_range, fill_value=0).reset_index()
full_counts.columns = ["hour", "count"]

x = full_counts["hour"].astype("int64") / 1e9
y = full_counts["count"]
poly_fit = Polynomial.fit(x, y, deg=6)

# Generate x values for the fit line
x_fit = np.linspace(x.min(), x.max(), 100)

# Evaluate the fitted polynomial at x_fit
y_fit = poly_fit(x_fit)

# Convert x_fit back to datetime
dates_fit = pd.to_datetime(x_fit * 1e9)  # reverse the earlier scaling


fig, ax = plt.subplots()
ax.scatter(full_counts["hour"], full_counts["count"])
ax.plot(dates_fit, y_fit, color="red", label="Polyfit (6th deg)")
time_fmt = mdates.DateFormatter("%m/%d %H:%M")
ax.xaxis.set_major_formatter(time_fmt)
plt.xticks(rotation=60)
plt.xlabel("Hour")
plt.ylabel("Number of Departures")
plt.title("Departures from Sydney Airport")
plt.legend()
plt.tight_layout()

fig2, ax2 = plt.subplots(figsize=(14, 6))
ax2.bar(arrival["arrival.iataCode"], arrival["count"])
ax2.set_xticks(arrival["arrival.iataCode"])
ax2.set_xticklabels(arrival["arrival.iataCode"], rotation=60)
ax2.set_xlabel("Arrival Airport")
ax2.set_ylabel("Number of Departures")
ax2.set_title("Departures from Sydney Airport")

fig3, ax3 = plt.subplots(figsize=(14, 6))
ax3.bar(airlines["airline.icaoCode"], airlines["count"])
ax3.set_xticks(airlines["airline.icaoCode"])
ax3.set_xticklabels(airlines["airline.icaoCode"], rotation=60)
ax3.set_xlabel("Airline (excluding less than 1 flight)")
ax3.set_ylabel("Number of Departures")
ax3.set_title("Airlines by Departure Numbers from Sydney Airport")

print(f"Total departures: {len(df)}")
plt.show()
