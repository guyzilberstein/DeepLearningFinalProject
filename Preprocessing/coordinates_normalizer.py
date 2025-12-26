import pandas as pd
import numpy as np

# 1. Load your data
df = pd.read_csv('your_collected_data.csv') # Assuming columns: 'filename', 'lat', 'lon'

# 2. Define the "Center" of your campus area (The Reference Point)
# A good trick is to just use the mean of your dataset
ref_lat = df['lat'].mean()
ref_lon = df['lon'].mean()

print(f"Reference Point: {ref_lat}, {ref_lon}")

# 3. Conversion Constants (Approximate for small areas)
# 1 degree of latitude is ~111,132 meters
# 1 degree of longitude depends on the latitude (cos(lat) * 111,132)
METERS_PER_LAT = 111132.0
METERS_PER_LON = 111132.0 * np.cos(np.radians(ref_lat))

# 4. Calculate local coordinates (Meters from Center)
df['x_meters'] = (df['lon'] - ref_lon) * METERS_PER_LON
df['y_meters'] = (df['lat'] - ref_lat) * METERS_PER_LAT

# 5. Save the prepared data
df.to_csv('processed_data.csv', index=False)
print("Preprocessing complete. Labels are now in meters!")
print(df[['x_meters', 'y_meters']].head())