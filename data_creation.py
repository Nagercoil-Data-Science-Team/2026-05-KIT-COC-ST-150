# ============================================================
# STEP 1 — INDUSTRIAL CO₂ DATASET PROCESSING
# FIXED VERSION
# ============================================================

# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ============================================================
# LOAD EXCEL FILE
# ============================================================

df = pd.read_excel("carbon data.xlsx")

# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

# Remove extra spaces
df.columns = df.columns.str.strip()

# Print column names
print("\nColumn Names:")
print(df.columns)

# ============================================================
# DISPLAY FIRST ROWS
# ============================================================

print("\nDataset Preview:")
print(df.head())

# ============================================================
# CHECK ACTUAL COLUMN NAMES
# ============================================================

# Rename columns if necessary
# (Modify based on printed column names)

df.rename(columns={
    'Primary Fuel ': 'Primary Fuel',
    'CO2 emission (Mton/yr)': 'CO2 emission'
}, inplace=True)

# ============================================================
# 1. LABEL ENCODING
# ============================================================

# Create encoders
sector_encoder = LabelEncoder()
fuel_encoder = LabelEncoder()
unit_encoder = LabelEncoder()

# Encode columns
df['Sector_Encoded'] = sector_encoder.fit_transform(df['Sector'])

df['Fuel_Encoded'] = fuel_encoder.fit_transform(df['Primary Fuel'])

df['Unit_Encoded'] = unit_encoder.fit_transform(df['Unit Type'])

# ============================================================
# PRINT ENCODING MAPPING
# ============================================================

print("\nFuel Encoding:")
for i, label in enumerate(fuel_encoder.classes_):
    print(label, "=", i)

# ============================================================
# 2. NORMALIZE CO₂ EMISSION
# ============================================================

scaler = MinMaxScaler()

df['CO2_Normalized'] = scaler.fit_transform(
    df[['CO2 emission']]
)

# ============================================================
# 3. SPATIAL FEATURES
# ============================================================

# ------------------------------------------------------------
# Emission Density
# ------------------------------------------------------------

facility_count = df.groupby('Province')['Facility ID'].transform('count')

province_emission = df.groupby('Province')['CO2 emission'].transform('sum')

df['Emission_Density'] = (
    province_emission / facility_count
)

# ------------------------------------------------------------
# Distance from Riyadh
# ------------------------------------------------------------

riyadh_lat = 24.7136
riyadh_lon = 46.6753

def haversine(lat1, lon1, lat2, lon2):

    R = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2)**2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2)**2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

df['Distance_From_Riyadh_km'] = haversine(
    df['Latitude'],
    df['Longitude'],
    riyadh_lat,
    riyadh_lon
)

# ------------------------------------------------------------
# Latitude-Longitude Interaction
# ------------------------------------------------------------

df['Lat_Long_Interaction'] = (
    df['Latitude'] * df['Longitude']
)

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\nProcessed Dataset:")
print(df.head())

# ============================================================
# SAVE OUTPUT
# ============================================================

df.to_excel(
    "processed_industrial_co2_data.xlsx",
    index=False
)

print("\nProcessed dataset saved successfully.")

# ============================================================
# STEP 2 — MICROBIAL DATASET PROCESSING
# (CSV FILE VERSION)
# ============================================================

# Import libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# LOAD CSV DATASET
# ============================================================

# Replace with your CSV file name
df = pd.read_csv("Dataset_GCB-22-1743.csv")

# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

# Remove extra spaces
df.columns = df.columns.str.strip()

# ============================================================
# DISPLAY COLUMN NAMES
# ============================================================

print("\nColumn Names:")
print(df.columns)

# ============================================================
# DISPLAY DATASET
# ============================================================

print("\nOriginal Dataset:")
print(df.head())

# ============================================================
# 1. SELECT IMPORTANT FEATURES
# ============================================================

selected_features = [
    'CO2 fixation rate',
    'MAT',
    'MAP',
    'pH',
    'SOC',
    'DOC',
    'MBC'
]

# Create dataframe with selected features
microbial_df = df[selected_features]

# ============================================================
# DISPLAY SELECTED DATA
# ============================================================

print("\nSelected Features:")
print(microbial_df.head())

# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:")
print(microbial_df.isnull().sum())

# ============================================================
# HANDLE MISSING VALUES
# ============================================================

# Fill missing values using mean
microbial_df = microbial_df.fillna(
    microbial_df.mean(numeric_only=True)
)

# ============================================================
# 2. CORRELATION ANALYSIS
# ============================================================

# Pearson correlation matrix
correlation_matrix = microbial_df.corr(method='pearson')

# Display correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# ============================================================
# HEATMAP VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title(
    'Microbial Feature Correlation Heatmap',
    fontsize=14,
    fontweight='bold'
)

plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()

plt.show()

# ============================================================
# 3. FEATURE SCALING
# ============================================================

# Initialize scaler
scaler = MinMaxScaler()

# Normalize features
scaled_values = scaler.fit_transform(microbial_df)

# Create normalized dataframe
scaled_df = pd.DataFrame(
    scaled_values,
    columns=microbial_df.columns
)

# ============================================================
# DISPLAY NORMALIZED DATA
# ============================================================

print("\nNormalized Dataset:")
print(scaled_df.head())

# ============================================================
# SAVE OUTPUT CSV FILE
# ============================================================

scaled_df.to_csv(
    "processed_microbial_data.csv",
    index=False
)

print("\nProcessed microbial dataset saved successfully.")

# ============================================================
# STEP 3 — ENVIRONMENTAL DATASET PROCESSING
# FIXED VERSION FOR YOUR DATA FORMAT
# ============================================================

# Import libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

# ============================================================
# LOAD EXCEL DATASET
# ============================================================

# Replace with your file name
df = pd.read_excel("environmental analysis.xlsx")

# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

# ============================================================
# DISPLAY COLUMN NAMES
# ============================================================

print("\nColumn Names:")
print(df.columns)

# ============================================================
# DISPLAY ORIGINAL DATASET
# ============================================================

print("\nOriginal Dataset:")
print(df.head())

# ============================================================
# 1. CREATE DATE COLUMN
# ============================================================

# Combine YEAR + MO + DY into Date
df['Date'] = pd.to_datetime({
    'year': df['YEAR'],
    'month': df['MO'],
    'day': df['DY']
})

# ============================================================
# DISPLAY DATE COLUMN
# ============================================================

print("\nDataset with Date Column:")
print(df[['YEAR', 'MO', 'DY', 'Date']].head())

# ============================================================
# 2. MONTHLY DATA AGGREGATION
# ============================================================

# Set Date as index
df.set_index('Date', inplace=True)

# Monthly average aggregation
monthly_df = df.resample('ME').mean(numeric_only=True)

# Reset index
monthly_df.reset_index(inplace=True)

# ============================================================
# DISPLAY MONTHLY DATA
# ============================================================

print("\nMonthly Aggregated Dataset:")
print(monthly_df.head())

# ============================================================
# 3. CREATE ENVIRONMENTAL INDICES
# ============================================================

# ------------------------------------------------------------
# Solar Efficiency Index (SEI)
# ------------------------------------------------------------

monthly_df['SEI'] = (
    monthly_df['ALLSKY_SFC_SW_DWN']
    * monthly_df['T2M']
)

# ------------------------------------------------------------
# Wind-Humidity Interaction
# ------------------------------------------------------------

monthly_df['Wind_Humidity_Index'] = (
    monthly_df['RH2M']
    / (monthly_df['WS10M'] + 1)
)

# ------------------------------------------------------------
# Environmental Suitability Score
# ------------------------------------------------------------

monthly_df['Environmental_Suitability'] = (
    monthly_df['SEI']
    + monthly_df['RH2M']
    - monthly_df['WS10M']
)

# ============================================================
# DISPLAY NEW FEATURES
# ============================================================

print("\nEnvironmental Features:")
print(
    monthly_df[
        [
            'SEI',
            'Wind_Humidity_Index',
            'Environmental_Suitability'
        ]
    ].head()
)

# ============================================================
# 4. FEATURE NORMALIZATION
# ============================================================

scaler = MinMaxScaler()

columns_to_scale = [
    'ALLSKY_SFC_SW_DWN',
    'T2M',
    'WS10M',
    'RH2M',
    'SEI',
    'Wind_Humidity_Index',
    'Environmental_Suitability'
]

monthly_df[columns_to_scale] = scaler.fit_transform(
    monthly_df[columns_to_scale]
)

# ============================================================
# DISPLAY FINAL DATASET
# ============================================================

print("\nProcessed Environmental Dataset:")
print(monthly_df.head())

# ============================================================
# SAVE OUTPUT TO EXCEL
# ============================================================

monthly_df.to_excel(
    "processed_environmental_data.xlsx",
    index=False
)

print("\nProcessed environmental dataset saved successfully.")

# ============================================================
# STEP 4 — DATA INTEGRATION
# FIXED VERSION
# INDUSTRIAL + MICROBIAL + ENVIRONMENTAL
# ============================================================

# Import libraries
import pandas as pd
import numpy as np

# ============================================================
# LOAD DATASETS
# ============================================================

industrial_df = pd.read_excel(
    "processed_industrial_co2_data.xlsx"
)

microbial_df = pd.read_csv(
    "processed_microbial_data.csv"
)

environment_df = pd.read_excel(
    "processed_environmental_data.xlsx"
)

# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

industrial_df.columns = industrial_df.columns.str.strip()
microbial_df.columns = microbial_df.columns.str.strip()
environment_df.columns = environment_df.columns.str.strip()

# ============================================================
# DISPLAY SHAPES
# ============================================================

print("\nIndustrial Shape:", industrial_df.shape)
print("Microbial Shape:", microbial_df.shape)
print("Environmental Shape:", environment_df.shape)

# ============================================================
# RESET INDEX
# ============================================================

industrial_df = industrial_df.reset_index(drop=True)
microbial_df = microbial_df.reset_index(drop=True)
environment_df = environment_df.reset_index(drop=True)

# ============================================================
# MATCH DATASET LENGTHS
# ============================================================

n_industrial = len(industrial_df)

# ------------------------------------------------------------
# Repeat environmental rows if needed
# ------------------------------------------------------------

environment_expanded = pd.concat(
    [environment_df] * (
        n_industrial // len(environment_df) + 1
    ),
    ignore_index=True
)

environment_expanded = environment_expanded.iloc[
    :n_industrial
]

# ------------------------------------------------------------
# Repeat microbial rows if needed
# ------------------------------------------------------------

microbial_expanded = pd.concat(
    [microbial_df] * (
        n_industrial // len(microbial_df) + 1
    ),
    ignore_index=True
)

microbial_expanded = microbial_expanded.iloc[
    :n_industrial
]

# ============================================================
# ADD ENVIRONMENTAL FEATURES
# ============================================================

industrial_df['Temperature'] = (
    environment_expanded['T2M']
)

industrial_df['Solar_Radiation'] = (
    environment_expanded['ALLSKY_SFC_SW_DWN']
)

industrial_df['Humidity'] = (
    environment_expanded['RH2M']
)

industrial_df['Wind_Speed'] = (
    environment_expanded['WS10M']
)

industrial_df['SEI'] = (
    environment_expanded['SEI']
)

industrial_df['Environmental_Suitability'] = (
    environment_expanded['Environmental_Suitability']
)

# ============================================================
# ADD MICROBIAL FEATURES
# ============================================================

industrial_df['CO2_Fixation_Rate'] = (
    microbial_expanded['CO2 fixation rate']
)

industrial_df['MAT'] = (
    microbial_expanded['MAT']
)

industrial_df['MAP'] = (
    microbial_expanded['MAP']
)

industrial_df['pH'] = (
    microbial_expanded['pH']
)

industrial_df['SOC'] = (
    microbial_expanded['SOC']
)

industrial_df['DOC'] = (
    microbial_expanded['DOC']
)

industrial_df['MBC'] = (
    microbial_expanded['MBC']
)

# ============================================================
# CREATE MICROBIAL SUITABILITY
# ============================================================

industrial_df['Microbial_Suitability'] = (
    0.4 * industrial_df['CO2_Fixation_Rate']
    + 0.3 * industrial_df['SOC']
    + 0.3 * industrial_df['MBC']
)

# ============================================================
# FINAL DATASET
# ============================================================

final_dataset = industrial_df[
    [
        'Facility ID',
        'City',
        'Province',
        'Latitude',
        'Longitude',
        'CO2 emission',
        'Primary Fuel',
        'Temperature',
        'Solar_Radiation',
        'Humidity',
        'Wind_Speed',
        'SEI',
        'Environmental_Suitability',
        'CO2_Fixation_Rate',
        'MAT',
        'MAP',
        'pH',
        'SOC',
        'DOC',
        'MBC',
        'Microbial_Suitability'
    ]
]

# ============================================================
# CHECK FEATURE VARIATION
# ============================================================

print("\nFeature Variability Check:")
print(
    final_dataset[
        [
            'Temperature',
            'Solar_Radiation',
            'Humidity',
            'Wind_Speed',
            'CO2_Fixation_Rate',
            'SOC',
            'DOC',
            'MBC'
        ]
    ].describe()
)

# ============================================================
# SAVE DATASET
# ============================================================

final_dataset.to_csv(
    "integrated_ai_input_dataset.csv",
    index=False
)

print("\nIntegrated dataset saved successfully.")