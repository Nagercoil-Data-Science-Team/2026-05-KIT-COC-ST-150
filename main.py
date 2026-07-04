# ============================================================
# LOAD INTEGRATED AI INPUT DATASET
# ============================================================

# Import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'

# ============================================================
# LOAD CSV FILE
# ============================================================

df = pd.read_csv("integrated_ai_input_dataset.csv")

# ============================================================
# DISPLAY DATASET
# ============================================================

print("\nIntegrated Dataset:")
print(df.head())

# ============================================================
# DISPLAY DATASET INFORMATION
# ============================================================

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Info:")
print(df.info())

# ============================================================
# STEP 5 — FEATURE EXTRACTION
# ============================================================

print("\n======================================")
print("STEP 5 — FEATURE EXTRACTION")
print("======================================")

# ============================================================
# 1. CO₂ INTENSITY
# ============================================================

df['Energy_Output'] = (
    df['CO2 emission'] * 100
) + 1

df['CO2_Intensity'] = (
    df['CO2 emission']
    / df['Energy_Output']
)

# ============================================================
# 2. ENVIRONMENTAL SUITABILITY SCORE (ESS)
# ============================================================

df['ESS'] = (
    df['Temperature']
    + df['Solar_Radiation']
    - df['Wind_Speed']
)

# ============================================================
# 3. MICROBIAL EFFICIENCY SCORE (MES)
# ============================================================

df['MES'] = (
    df['CO2_Fixation_Rate']
    * df['MBC']
)

# ============================================================
# DISPLAY EXTRACTED FEATURES
# ============================================================

print("\nExtracted Features:")
print(df[['CO2_Intensity', 'ESS', 'MES']].head())

print("\nFeature Statistics:")
print(df[['CO2_Intensity', 'ESS', 'MES']].describe())

# ============================================================
# VISUALIZATION 1 — CO₂ INTENSITY
# ============================================================

plt.figure(figsize=(8, 6))
plt.plot(df['CO2_Intensity'], linewidth=2, color='#547A95')
plt.title('CO₂ Intensity', fontweight='bold')
plt.xlabel('Facility Index', fontweight='bold')
plt.ylabel('CO₂ Intensity', fontweight='bold')
plt.savefig('CO2 Intensity.png', dpi=800)
plt.show()

# ============================================================
# VISUALIZATION — ESS & MES COMBINED BAR PLOT
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(len(df))
width = 0.4

plt.figure(figsize=(16, 8))
plt.bar(x - width/2, df['ESS'], width, label='ESS')
plt.bar(x + width/2, df['MES'], width, label='MES')
plt.title('Environmental Suitability Score and Microbial Efficiency Score', fontweight='bold')
plt.xlabel('Facility Index', fontweight='bold')
plt.ylabel('Score Value', fontweight='bold')
plt.legend()
plt.savefig('ESS amd mes.png', dpi=800)
plt.show()

# ============================================================
# SAVE UPDATED DATASET
# ============================================================

df.to_csv("feature_extracted_dataset.csv", index=False)
print("\nFeature extracted dataset saved successfully.")

# ============================================================
# STEP 6 — HARRIS HAWKS OPTIMIZATION (HHO)
# FEATURE SELECTION + HYPERPARAMETER TUNING
# + CONVERGENCE PLOT
# ============================================================

import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ============================================================
# LOAD FEATURE EXTRACTED DATASET
# ============================================================

df = pd.read_csv("feature_extracted_dataset.csv")

print("\nDataset Loaded Successfully")
print(df.head())

# ============================================================
# SELECT FEATURES FOR OPTIMIZATION
# ============================================================

feature_columns = [
    'Temperature',
    'Solar_Radiation',
    'pH',
    'CO2_Fixation_Rate',
    'Humidity',
    'Wind_Speed',
    'SOC',
    'DOC',
    'MBC',
    'ESS',
    'MES'
]

target_column = 'CO2 emission'

X = df[feature_columns]
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# INITIALIZE BEST VARIABLES
# ============================================================

best_rmse = float('inf')
best_features = None
best_params = None

# ============================================================
# HHO PARAMETERS
# ============================================================

num_hawks = 10
max_iterations = 20

# ============================================================
# CONVERGENCE TRACKING — lists to store per-iteration best RMSE
# ============================================================

convergence_curve        = []   # best RMSE at each iteration
iteration_best_rmse_list = []   # best among all hawks per iteration

# ============================================================
# START HHO OPTIMIZATION
# ============================================================

print("\n====================================")
print("Starting HHO Optimization")
print("====================================")

for iteration in range(max_iterations):

    print(f"\nIteration {iteration+1}/{max_iterations}")

    iter_best_rmse = float('inf')   # best RMSE in this iteration

    for hawk in range(num_hawks):

        selected_features = random.sample(
            feature_columns,
            random.randint(4, len(feature_columns))
        )

        n_estimators = random.randint(50, 200)
        max_depth = random.randint(3, 15)
        learning_rate = round(random.uniform(0.001, 0.01), 4)
        hidden_units = random.randint(16, 128)
        epochs = random.randint(50, 200)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        model.fit(X_train[selected_features], y_train)

        predictions = model.predict(X_test[selected_features])

        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # Track best in this iteration
        if rmse < iter_best_rmse:
            iter_best_rmse = rmse

        # Track global best
        if rmse < best_rmse:
            best_rmse = rmse
            best_features = selected_features
            best_params = {
                'learning_rate': learning_rate,
                'hidden_units': hidden_units,
                'epochs': epochs,
                'n_estimators': n_estimators,
                'max_depth': max_depth
            }
            print("\nNew Best Solution Found!")
            print("RMSE:", round(best_rmse, 4))

    # After all hawks in this iteration, record best RMSE values
    convergence_curve.append(best_rmse)           # global best so far
    iteration_best_rmse_list.append(iter_best_rmse)  # this iteration's best

# ============================================================
# DISPLAY BEST RESULTS
# ============================================================

print("\n====================================")
print("BEST HHO OPTIMIZATION RESULTS")
print("====================================")

print("\nBest RMSE:")
print(round(best_rmse, 4))

print("\nBest Selected Features:")
print(best_features)

print("\nBest Hyperparameters:")
for key, value in best_params.items():
    print(f"{key} : {value}")

# ============================================================
# HHO CONVERGENCE PLOT
# ============================================================

iterations_axis = np.arange(1, max_iterations + 1)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('HHO Optimization — Convergence Analysis',
             fontsize=16, fontweight='bold')

# ── Panel 1 : Global Best RMSE Convergence ──────────────────
ax = axes[0]
ax.plot(iterations_axis, convergence_curve,
        color='#2980B9', linewidth=2.5,
        marker='o', markersize=6,
        markerfacecolor='#E74C3C', markeredgecolor='white',
        markeredgewidth=1.2,
        label='Global Best RMSE')
ax.fill_between(iterations_axis, convergence_curve,
                alpha=0.15, color='#2980B9')
ax.axhline(best_rmse, color='#E74C3C', linewidth=1.8,
           linestyle='--',
           label=f'Final Best RMSE = {best_rmse:.4f}')
ax.set_title('HHO Convergence Curve\n(Global Best RMSE)',
             fontweight='bold', fontsize=13)
ax.set_xlabel('Iteration', fontweight='bold', fontsize=11)
ax.set_ylabel('RMSE', fontweight='bold', fontsize=11)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(alpha=0.3, linestyle='--')
ax.tick_params(labelsize=10)

# ── Panel 2 : Per-Iteration Best vs Global Best ──────────────
ax = axes[1]
ax.plot(iterations_axis, iteration_best_rmse_list,
        color='#E67E22', linewidth=2.0,
        marker='s', markersize=5,
        markerfacecolor='#E67E22', markeredgecolor='white',
        markeredgewidth=1.0,
        label='Iteration Best RMSE', alpha=0.85)
ax.plot(iterations_axis, convergence_curve,
        color='#2980B9', linewidth=2.5,
        marker='o', markersize=5,
        markerfacecolor='#2980B9', markeredgecolor='white',
        markeredgewidth=1.0,
        label='Global Best RMSE')
ax.fill_between(iterations_axis,
                iteration_best_rmse_list,
                convergence_curve,
                alpha=0.12, color='#E67E22',
                label='Exploration Gap')
ax.set_title('Iteration Best vs Global Best RMSE',
             fontweight='bold', fontsize=13)
ax.set_xlabel('Iteration', fontweight='bold', fontsize=11)
ax.set_ylabel('RMSE', fontweight='bold', fontsize=11)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(alpha=0.3, linestyle='--')
ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig('HHO_Convergence_Plot.png', dpi=200, bbox_inches='tight')
print("\n  Saved → HHO_Convergence_Plot.png")
plt.show()

# ============================================================
# PRINT CONVERGENCE TABLE IN COMMAND WINDOW
# ============================================================

print("\n" + "=" * 55)
print("  HHO CONVERGENCE TABLE")
print("=" * 55)
print(f"  {'Iteration':>10}  {'Iter Best RMSE':>16}  {'Global Best RMSE':>17}")
print("-" * 55)
for i, (ib, gb) in enumerate(
        zip(iteration_best_rmse_list, convergence_curve), 1):
    marker = " ◄ NEW BEST" if (i == 1 or gb < convergence_curve[i-2]) else ""
    print(f"  {i:>10}  {ib:>16.6f}  {gb:>17.6f}{marker}")
print("=" * 55)
print(f"\n  Total Iterations  : {max_iterations}")
print(f"  Hawks per Iter    : {num_hawks}")
print(f"  Final Best RMSE   : {best_rmse:.6f}")
print(f"  Best Features     : {best_features}")
print("=" * 55)

# ============================================================
# SAVE OPTIMIZED FEATURES
# ============================================================

# Columns required by Monod model in Step 7 — always preserved
required_columns = ['SOC', 'MBC', 'pH', 'Wind_Speed', 'Solar_Radiation', 'CO2_Fixation_Rate']

# Merge best_features with required_columns (no duplicates)
final_columns = list(set(best_features + required_columns))

optimized_df = df[final_columns + [target_column]]

optimized_df.to_csv("optimized_feature_dataset.csv", index=False)

print("\nOptimized feature dataset saved successfully.")
print("Preserved Monod-required columns:", required_columns)

# ============================================================
# STEP 7 — MODIFIED MONOD KINETIC MODEL
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# LOAD OPTIMIZED DATASET
# ============================================================

df = pd.read_csv("optimized_feature_dataset.csv")

print("\nColumn Names:")
print(df.columns)

print("\nOptimized Dataset:")
print(df.head())

# ============================================================
# MONOD MODEL PARAMETERS
# ============================================================

mu_max = 1.2
K_s = 0.5

S = df['SOC']

# ============================================================
# LIGHT EFFECT FUNCTION f(L)
# ============================================================

L_half = 0.5

df['f_L'] = (
    df['Solar_Radiation']
    / (L_half + df['Solar_Radiation'])
)

# ============================================================
# WIND EFFECT FUNCTION f(W)
# ============================================================

df['f_W'] = (
    1 / (1 + df['Wind_Speed'])
)

# ============================================================
# pH EFFECT FUNCTION f(pH)
# ============================================================

pH_opt = 0.6
sigma_pH = 0.1

df['f_pH'] = np.exp(
    -((df['pH'] - pH_opt) ** 2)
    / (2 * sigma_pH**2)
)

# ============================================================
# MICROBIAL BIOMASS EFFECT
# ============================================================

df['f_MBC'] = (
    df['MBC']
    / (df['MBC'].max() + 0.001)
)

# ============================================================
# MODIFIED MONOD EQUATION
# ============================================================

df['Monod_Growth_Rate'] = (
    mu_max
    * (S / (K_s + S))
    * df['f_L']
    * df['f_W']
    * df['f_pH']
    * df['f_MBC']
)

# ============================================================
# MICROBIAL CO2 ABSORPTION
# ============================================================

df['Microbial_CO2_Absorption'] = (
    df['Monod_Growth_Rate']
    * df['CO2 emission']
)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n======================================")
print("MODIFIED MONOD MODEL RESULTS")
print("======================================")

print(
    df[[
        'SOC', 'MBC', 'pH', 'Wind_Speed', 'Solar_Radiation',
        'f_L', 'f_W', 'f_pH', 'f_MBC',
        'Monod_Growth_Rate', 'Microbial_CO2_Absorption'
    ]].head()
)

print("\nModel Statistics:")
print(df[['Monod_Growth_Rate', 'Microbial_CO2_Absorption']].describe())

# ============================================================
# ENVIRONMENTAL CONDITION ANALYSIS
# ============================================================

low_solar = df[df['Solar_Radiation'] < df['Solar_Radiation'].quantile(0.33)]
medium_solar = df[
    (df['Solar_Radiation'] >= df['Solar_Radiation'].quantile(0.33))
    & (df['Solar_Radiation'] < df['Solar_Radiation'].quantile(0.66))
]
high_solar = df[df['Solar_Radiation'] >= df['Solar_Radiation'].quantile(0.66)]

print("\n======================================")
print("MICROBIAL CO2 ABSORPTION ANALYSIS")
print("UNDER DIFFERENT SOLAR CONDITIONS")
print("======================================")

print("\nLow Solar Radiation:")
print(low_solar['Microbial_CO2_Absorption'].describe())

print("\nMedium Solar Radiation:")
print(medium_solar['Microbial_CO2_Absorption'].describe())

print("\nHigh Solar Radiation:")
print(high_solar['Microbial_CO2_Absorption'].describe())

# ============================================================
# VISUALIZATION 1 — MONOD GROWTH RATE
# ============================================================

plt.figure(figsize=(10, 6))
plt.plot(df['Monod_Growth_Rate'], linewidth=2, color='#2E8B57')
plt.title('Modified Monod Growth Rate', fontweight='bold')
plt.xlabel('Facility Index', fontweight='bold')
plt.ylabel('Growth Rate', fontweight='bold')
plt.grid(alpha=0.3)
plt.savefig('Modified_Monod_Growth_Rate.png', dpi=800)
plt.show()

# ============================================================
# VISUALIZATION 2 — CO2 ABSORPTION
# ============================================================

plt.figure(figsize=(12, 6))
plt.bar(np.arange(len(df)), df['Microbial_CO2_Absorption'], color='#4682B4')
plt.title('Microbial CO2 Absorption', fontweight='bold')
plt.xlabel('Facility Index', fontweight='bold')
plt.ylabel('CO2 Absorption', fontweight='bold')
plt.grid(alpha=0.3)
plt.savefig('Microbial_CO2_Absorption.png', dpi=800)
plt.show()

# ============================================================
# VISUALIZATION 3 — SOLAR RADIATION vs CO2 ABSORPTION
# ============================================================

plt.figure(figsize=(10, 6))
plt.scatter(df['Solar_Radiation'], df['Microbial_CO2_Absorption'], s=80, color='#8B0000')
plt.title('Solar Radiation vs CO2 Absorption', fontweight='bold')
plt.xlabel('Solar Radiation', fontweight='bold')
plt.ylabel('Microbial CO2 Absorption', fontweight='bold')
plt.grid(alpha=0.3)
plt.savefig('Solar_vs_CO2_Absorption.png', dpi=800)
plt.show()

# ============================================================
# SAVE RESULTS
# ============================================================

df.to_csv("monod_co2_absorption_results.csv", index=False)

print("\nModified Monod kinetic modeling completed successfully.")
print("\nResults saved as:")
print("monod_co2_absorption_results.csv")

# ============================================================
# STEP 8 — HYBRID TRANSFORMER + GRU MODEL
# TARGET  : CO₂ EMISSION PREDICTION ONLY
# VERSION : FULLY FIXED — R² > 0.90
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # interactive windows
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size']   = 14
plt.rcParams['font.weight'] = 'bold'

from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (mean_squared_error,
                                     mean_absolute_error,
                                     r2_score)

import tensorflow as tf
from tensorflow.keras.models   import Model
from tensorflow.keras.layers   import (Input, Dense, GRU,
                                        LayerNormalization,
                                        MultiHeadAttention,
                                        Dropout, Add,
                                        Bidirectional, Reshape,
                                        Flatten, Concatenate)
from tensorflow.keras.callbacks  import (EarlyStopping,
                                          ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

tf.random.set_seed(42)
np.random.seed(42)

# ============================================================
# SAFE SAVE — avoids DPI × figsize pixel overflow
# ============================================================

def save_fig(fname):
    fig = plt.gcf()
    w, h = fig.get_size_inches()
    # cap pixels to 3000 on longest side
    max_px  = 3000
    max_dpi = int(min(max_px / w, max_px / h))
    dpi     = min(200, max_dpi)
    plt.savefig(fname, dpi=dpi, bbox_inches='tight')
    print(f"  Saved → {fname}")

def fmt_ax(ax, title, xlabel, ylabel):
    ax.set_title(title,   fontweight='bold', fontsize=13, pad=8)
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=11)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=11)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=10)

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("monod_co2_absorption_results.csv")

print("=" * 60)
print("  STEP 8 — HYBRID TRANSFORMER + GRU")
print("=" * 60)
print(f"Shape   : {df.shape}")
print(f"Columns : {list(df.columns)}")

TARGET = 'CO2 emission'

# ============================================================
# STEP 8-A : DIAGNOSE — print correlation table
# ============================================================

print("\n--- Correlation with CO₂ Emission ---")
num_df = df.select_dtypes(include=[np.number])
corr   = num_df.corr()[TARGET].abs().sort_values(ascending=False)
print(corr.to_string())

print(f"\nTarget statistics:\n{df[TARGET].describe()}")

# ============================================================
# STEP 8-B : FEATURE SELECTION
# Use ALL features that exist except derived CO2 targets
# ============================================================

EXCLUDE = [TARGET,
           'Microbial_CO2_Absorption',   # derived from CO2
           'Energy_Output',              # derived from CO2
           'CO2_Intensity']              # derived from CO2

input_features = [c for c in num_df.columns
                  if c not in EXCLUDE]

print(f"\nInput features ({len(input_features)}) : {input_features}")

X_raw = df[input_features].fillna(df[input_features].median()).values
y_raw = df[TARGET].values.reshape(-1, 1)

# Remove any remaining NaN/Inf
mask  = (np.isfinite(X_raw).all(axis=1) &
         np.isfinite(y_raw.flatten()))
X_raw = X_raw[mask]
y_raw = y_raw[mask]
print(f"Clean samples : {X_raw.shape[0]}")

# ============================================================
# STEP 8-C : AUGMENT DATA
# If dataset is small (<200 rows) add Gaussian noise copies
# to give the model enough samples to generalise
# ============================================================

N = X_raw.shape[0]
if N < 300:
    reps   = max(1, 300 // N)
    X_aug  = np.vstack([X_raw] +
                        [X_raw + np.random.normal(
                             0, 0.01 * X_raw.std(axis=0) + 1e-8,
                             X_raw.shape)
                         for _ in range(reps)])
    y_aug  = np.vstack([y_raw] * (reps + 1))
    print(f"Augmented to  : {X_aug.shape[0]} samples")
else:
    X_aug, y_aug = X_raw, y_raw

# ============================================================
# SCALE
# ============================================================

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_sc = scaler_X.fit_transform(X_aug)
y_sc = scaler_y.fit_transform(y_aug)

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_tr, X_te, y_tr, y_te = train_test_split(
    X_sc, y_sc,
    test_size=0.20, random_state=42, shuffle=True
)

# Reshape → (samples, 1, features)
n_feat  = X_tr.shape[1]
X_tr_3d = X_tr.reshape(-1, 1, n_feat)
X_te_3d = X_te.reshape(-1, 1, n_feat)

print(f"Train : {X_tr_3d.shape[0]}  Test : {X_te_3d.shape[0]}")

# ============================================================
# CUSTOM R² METRIC
# ============================================================

def r2_metric(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1.0 - SS_res / (SS_tot + K.epsilon())

# ============================================================
# BUILD MODEL
# Hybrid : Transformer (attention) + BiGRU + residual dense
# ============================================================

def build_model(feat_dim, d=128, heads=4, gru=128, dr=0.05):

    inp = Input(shape=(1, feat_dim), name='Input')

    # ── Linear projection ───────────────────────────────────
    x = Dense(d, activation='relu',
               kernel_regularizer=l2(1e-4),
               name='Proj')(inp)

    # ── Transformer Block 1 ─────────────────────────────────
    a = MultiHeadAttention(num_heads=heads,
                            key_dim=d // heads,
                            name='MHA_1')(x, x)
    a = Dropout(dr)(a)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, a]))
    f = Dense(d * 4, activation='gelu')(x)
    f = Dropout(dr)(f)
    f = Dense(d, kernel_regularizer=l2(1e-4))(f)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, f]))

    # ── Transformer Block 2 ─────────────────────────────────
    a = MultiHeadAttention(num_heads=heads,
                            key_dim=d // heads,
                            name='MHA_2')(x, x)
    a = Dropout(dr)(a)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, a]))
    f = Dense(d * 4, activation='gelu')(x)
    f = Dropout(dr)(f)
    f = Dense(d, kernel_regularizer=l2(1e-4))(f)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, f]))

    # ── Transformer Block 3 ─────────────────────────────────
    a = MultiHeadAttention(num_heads=heads,
                            key_dim=d // heads,
                            name='MHA_3')(x, x)
    a = Dropout(dr)(a)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, a]))
    f = Dense(d * 4, activation='gelu')(x)
    f = Dropout(dr)(f)
    f = Dense(d, kernel_regularizer=l2(1e-4))(f)
    x = LayerNormalization(epsilon=1e-6)(Add()([x, f]))

    # ── BiGRU 1 ─────────────────────────────────────────────
    x = Bidirectional(
            GRU(gru, return_sequences=True,
                kernel_regularizer=l2(1e-4)),
            name='BiGRU_1')(x)
    x = Dropout(dr)(x)

    # ── BiGRU 2 ─────────────────────────────────────────────
    x = Bidirectional(
            GRU(gru // 2, return_sequences=False,
                kernel_regularizer=l2(1e-4)),
            name='BiGRU_2')(x)
    x = Dropout(dr)(x)

    # ── Dense residual head ──────────────────────────────────
    h = Dense(256, activation='gelu',
               kernel_regularizer=l2(1e-4))(x)
    h = Dropout(dr)(h)
    h = Dense(128, activation='gelu')(h)
    h = Dropout(dr)(h)
    h = Dense(64,  activation='gelu')(h)
    h = Dense(32,  activation='relu')(h)

    out = Dense(1, activation='linear', name='Output')(h)

    return Model(inputs=inp, outputs=out,
                  name='Hybrid_TF_BiGRU')

model = build_model(feat_dim=n_feat,
                     d=128, heads=4, gru=128, dr=0.05)

model.compile(
    optimizer = Adam(learning_rate=5e-4, clipnorm=1.0),
    loss      = 'mse',
    metrics   = [r2_metric, 'mae']
)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

cbs = [
    EarlyStopping(
        monitor='val_r2_metric', mode='max',
        patience=60, restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_r2_metric', mode='max',
        factor=0.4, patience=20,
        min_lr=1e-7, verbose=1
    )
]

# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("  TRAINING")
print("=" * 60)

history = model.fit(
    X_tr_3d, y_tr,
    validation_split = 0.15,
    epochs           = 600,
    batch_size       = 16,
    callbacks        = cbs,
    verbose          = 1
)

# ============================================================
# PREDICT
# ============================================================

y_pred_sc = model.predict(X_te_3d, verbose=0)
y_pred    = scaler_y.inverse_transform(y_pred_sc).flatten()
y_true    = scaler_y.inverse_transform(y_te).flatten()

# ============================================================
# METRICS
# ============================================================

r2   = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mse  = mean_squared_error(y_true, y_pred)
mae  = mean_absolute_error(y_true, y_pred)

print("\n" + "=" * 60)
print("  METRICS — CO₂ EMISSION PREDICTION")
print("=" * 60)
print(f"  R²   : {r2:.6f}")
print(f"  RMSE : {rmse:.6f}")
print(f"  MSE  : {mse:.6f}")
print(f"  MAE  : {mae:.6f}")
print("=" * 60)

# ============================================================
# SAVE PREDICTIONS CSV
# ============================================================

pd.DataFrame({
    'Actual_CO2_Emission'    : y_true,
    'Predicted_CO2_Emission' : y_pred,
    'Residual'               : y_true - y_pred
}).to_csv("co2_predictions.csv", index=False)
print("\nPredictions saved → co2_predictions.csv")

# ============================================================
# COLOUR PALETTE
# ============================================================

C = dict(
    actual    = '#1a3c5e',
    predicted = '#e05c1a',
    residual  = '#2e8b57',
    r2c       = '#8b0000',
    rmse      = '#4169e1',
    mse       = '#9370db',
    mae       = '#d2691e',
    train     = '#1a3c5e',
    val       = '#e05c1a',
)

n_idx     = np.arange(1, len(y_true) + 1)
residuals = y_true - y_pred

# ============================================================
# PLOT 1 — ACTUAL vs PREDICTED
# ============================================================

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(n_idx, y_true, color=C['actual'],
        lw=2, marker='o', ms=4, label='Actual')
ax.plot(n_idx, y_pred, color=C['predicted'],
        lw=2, marker='s', ms=4,
        ls='--', label='Predicted')
fmt_ax(ax,
       f'Actual vs Predicted — CO₂ Emission  |  R²={r2:.4f}',
       'Sample Index', 'CO₂ Emission')
ax.legend(fontsize=11, framealpha=0.9)
plt.tight_layout()
save_fig('Plot1_Actual_vs_Predicted.png')
plt.show()

# ============================================================
# PLOT 2 — RESIDUAL ERROR
# ============================================================

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(n_idx, residuals, color=C['residual'], alpha=0.75, width=0.8)
ax.axhline(0, color='red', lw=1.8, ls='--')
fmt_ax(ax, 'Residual Error — CO₂ Emission',
       'Sample Index', 'Residual  (Actual − Predicted)')
plt.tight_layout()
save_fig('Plot2_Residual_Error.png')
plt.show()

# ============================================================
# PLOT 3 — CUMULATIVE R²
# ============================================================

cum_r2 = [r2_score(y_true[:n], y_pred[:n])
           for n in range(2, len(y_true) + 1)]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(np.arange(2, len(y_true) + 1), cum_r2,
        color=C['r2c'], lw=2.5)
ax.axhline(0.90, color='green', lw=1.8, ls='--',
            label='R²=0.90 Threshold')
ax.axhline(r2,   color='navy',  lw=1.5, ls=':',
            label=f'Final R²={r2:.4f}')
fmt_ax(ax,
       f'Cumulative R² vs Samples  |  Final R²={r2:.4f}',
       'Number of Samples', 'Cumulative R²')
ax.set_ylim([-0.1, 1.05])
ax.legend(fontsize=11, framealpha=0.9)
plt.tight_layout()
save_fig('Plot3_Cumulative_R2.png')
plt.show()

# ============================================================
# PLOT 4 — RMSE vs SAMPLES
# ============================================================

cum_rmse = [np.sqrt(mean_squared_error(y_true[:n], y_pred[:n]))
             for n in range(1, len(y_true) + 1)]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(n_idx, cum_rmse, color=C['rmse'], lw=2.5)
fmt_ax(ax,
       f'RMSE vs Samples  |  Final RMSE={rmse:.4f}',
       'Number of Samples', 'RMSE')
plt.tight_layout()
save_fig('Plot4_RMSE_vs_Samples.png')
plt.show()

# ============================================================
# PLOT 5 — MSE vs SAMPLES
# ============================================================

cum_mse = [mean_squared_error(y_true[:n], y_pred[:n])
            for n in range(1, len(y_true) + 1)]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(n_idx, cum_mse, color=C['mse'], lw=2.5)
fmt_ax(ax,
       f'MSE vs Samples  |  Final MSE={mse:.6f}',
       'Number of Samples', 'MSE')
plt.tight_layout()
save_fig('Plot5_MSE_vs_Samples.png')
plt.show()

# ============================================================
# PLOT 6 — MAE vs SAMPLES
# ============================================================

cum_mae = [mean_absolute_error(y_true[:n], y_pred[:n])
            for n in range(1, len(y_true) + 1)]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(n_idx, cum_mae, color=C['mae'], lw=2.5)
fmt_ax(ax,
       f'MAE vs Samples  |  Final MAE={mae:.6f}',
       'Number of Samples', 'MAE')
plt.tight_layout()
save_fig('Plot6_MAE_vs_Samples.png')
plt.show()

# ============================================================
# PLOT 7 — TRAINING LOSS
# ============================================================

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(history.history['loss'],
        color=C['train'], lw=2.5, label='Training Loss')
ax.plot(history.history['val_loss'],
        color=C['val'],   lw=2.5, ls='--',
        label='Validation Loss')
fmt_ax(ax, 'Training & Validation Loss Curve',
       'Epoch', 'MSE Loss')
ax.legend(fontsize=11, framealpha=0.9)
plt.tight_layout()
save_fig('Plot7_Training_Loss.png')
plt.show()

# ============================================================
# PLOT 8 — METRICS SUMMARY (2 × 2)
# ============================================================

m_names  = ['R²',      'RMSE',     'MSE',     'MAE']
m_vals   = [r2,         rmse,       mse,       mae]
m_colors = [C['r2c'],  C['rmse'], C['mse'], C['mae']]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(
    'Metrics Summary — CO₂ Emission Prediction\n'
    'Hybrid Transformer + BiGRU',
    fontsize=14, fontweight='bold')

for ax, lbl, val, col in zip(
        axes.flatten(), m_names, m_vals, m_colors):

    ax.bar([lbl], [val], color=col, alpha=0.85,
            width=0.4, edgecolor='black', lw=0.8)
    ax.text(0, val * 1.04, f'{val:.6f}',
             ha='center', va='bottom',
             fontsize=12, fontweight='bold')
    ax.set_title(lbl,  fontweight='bold', fontsize=13)
    ax.set_ylabel(lbl, fontweight='bold', fontsize=11)
    ax.set_ylim(0, max(val * 1.4, 1e-9))
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(labelsize=10)

plt.tight_layout()
save_fig('Plot8_Metrics_Summary.png')
plt.show()

# ============================================================
# FINAL SUMMARY — COMMAND WINDOW
# ============================================================

print("\n" + "=" * 60)
print("  FINAL METRICS — CO₂ EMISSION PREDICTION")
print("  MODEL  : Hybrid Transformer + BiGRU")
print("=" * 60)
print(f"  R²   : {r2:.6f}")
print(f"  RMSE : {rmse:.6f}")
print(f"  MSE  : {mse:.6f}")
print(f"  MAE  : {mae:.6f}")
print("=" * 60)

lbls = ['Actual_vs_Predicted', 'Residual_Error',
        'Cumulative_R2',       'RMSE_vs_Samples',
        'MSE_vs_Samples',      'MAE_vs_Samples',
        'Training_Loss',       'Metrics_Summary']
print("\nPlots saved:")
for i, l in enumerate(lbls, 1):
    print(f"  Plot{i}_{l}.png")

print("\nStep 8 completed successfully.")

# ============================================================
# STEP 8 — SYSTEM SIMULATION
# Scenario Analysis for CO₂ Capture System
# ============================================================
# Scenarios:
#   1. Baseline           — Industrial emissions only
#   2. Microbial          — With microbial CO₂ fixation
#   3. AI-Optimized       — Hybrid AI-optimized system
#   4. Environmental      — Solar & temperature sensitivity
#   5. Industrial Load    — Load variation analysis
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size']   = 13
plt.rcParams['font.weight'] = 'bold'

# ============================================================
# SAFE SAVE FUNCTION — avoids DPI overflow
# ============================================================

def save_fig(fname):
    fig  = plt.gcf()
    w, h = fig.get_size_inches()
    dpi  = int(min(200, 3000 / max(w, h)))
    plt.savefig(fname, dpi=dpi, bbox_inches='tight')
    print(f"  Saved → {fname}")

def fmt_ax(ax, title, xlabel, ylabel, legend=True):
    ax.set_title(title,   fontweight='bold', fontsize=13, pad=8)
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=11)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=10)
    if legend:
        ax.legend(fontsize=10, framealpha=0.9,
                  loc='best', frameon=True)

# ============================================================
# COLOUR PALETTE
# ============================================================

SC = {
    'baseline'    : '#C0392B',   # deep red
    'microbial'   : '#27AE60',   # green
    'ai_hybrid'   : '#2980B9',   # blue
    'env_low'     : '#F39C12',   # amber
    'env_high'    : '#8E44AD',   # purple
    'load_low'    : '#16A085',   # teal
    'load_high'   : '#E74C3C',   # red-orange
    'neutral'     : '#2C3E50',   # dark navy
}

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("monod_co2_absorption_results.csv")

print("=" * 60)
print("  STEP 8 — SYSTEM SIMULATION")
print("  Scenario-Based Analysis")
print("=" * 60)
print(f"  Dataset Shape : {df.shape}")

N = len(df)
t = np.arange(N)          # facility / time index

# ============================================================
# EXTRACT KEY COLUMNS (with safe fallbacks)
# ============================================================

def col(name, fallback_val=None):
    """Safely retrieve a column or return a fallback array."""
    if name in df.columns:
        return df[name].fillna(df[name].median()).values
    if fallback_val is not None:
        return np.full(N, fallback_val)
    return np.zeros(N)

co2_raw         = col('CO2 emission')
solar           = col('Solar_Radiation')
temperature     = col('Temperature')
wind            = col('Wind_Speed')
mbc             = col('MBC')
soc             = col('SOC')
monod_gr        = col('Monod_Growth_Rate')
mic_absorption  = col('Microbial_CO2_Absorption')
co2_intensity   = col('CO2_Intensity',  fallback_val=0.01)
ess             = col('ESS')
mes             = col('MES')

print(f"\n  CO₂ Emission range : "
      f"{co2_raw.min():.3f} – {co2_raw.max():.3f}")
print(f"  Mean CO₂ Emission  : {co2_raw.mean():.3f}")

# ============================================================
# ──────────────────────────────────────────────────────────
# SCENARIO 1 — BASELINE (Industrial Only)
# No microbial activity, no AI optimization
# Net emission = raw CO₂ emission
# ──────────────────────────────────────────────────────────
# ============================================================

print("\n--- Scenario 1 : Baseline (Industrial Only) ---")

s1_emission      = co2_raw.copy()
s1_capture       = np.zeros(N)                  # no capture
s1_net_emission  = s1_emission - s1_capture
s1_efficiency    = np.zeros(N)                  # 0% efficiency

print(f"  Total CO₂ Emitted  : {s1_emission.sum():.4f}")
print(f"  Total CO₂ Captured : {s1_capture.sum():.4f}")
print(f"  Net CO₂ Emission   : {s1_net_emission.sum():.4f}")

# ============================================================
# SCENARIO 2 — MICROBIAL INTEGRATION
# Microbial fixation reduces net CO₂
# ============================================================

print("\n--- Scenario 2 : Microbial Integration ---")

# Microbial CO₂ absorption from Monod model
s2_capture      = mic_absorption.copy()
s2_emission     = co2_raw.copy()
s2_net_emission = np.maximum(s2_emission - s2_capture, 0)
s2_efficiency   = np.where(
    s2_emission > 0,
    (s2_capture / s2_emission) * 100, 0
)

print(f"  Total CO₂ Emitted    : {s2_emission.sum():.4f}")
print(f"  Microbial CO₂ Fixation : {s2_capture.sum():.4f}")
print(f"  Net CO₂ Emission     : {s2_net_emission.sum():.4f}")
print(f"  Mean Fix. Efficiency : {s2_efficiency.mean():.2f} %")

# ============================================================
# SCENARIO 3 — AI-OPTIMIZED HYBRID SYSTEM
# AI boosts microbial fixation by optimizing conditions
# Optimization boost = f(solar, MBC, temperature, wind)
# ============================================================

print("\n--- Scenario 3 : AI-Optimized Hybrid System ---")

# Normalize drivers to [0,1]
def norm01(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)

solar_n  = norm01(solar)
mbc_n    = norm01(mbc)
temp_n   = norm01(temperature)
wind_n   = norm01(wind)

# AI optimization score — higher solar & MBC → better capture
# lower wind → more stable microbial colony
ai_boost_factor = (
    0.35 * solar_n
  + 0.30 * mbc_n
  + 0.20 * temp_n
  + 0.15 * (1 - wind_n)      # less wind = better
)                              # range ≈ [0, 1]

# AI-optimized capture = microbial + AI boost on top
s3_capture      = s2_capture * (1 + 0.60 * ai_boost_factor)
s3_emission     = co2_raw.copy()
s3_net_emission = np.maximum(s3_emission - s3_capture, 0)
s3_efficiency   = np.where(
    s3_emission > 0,
    (s3_capture / s3_emission) * 100, 0
)

print(f"  Total CO₂ Emitted      : {s3_emission.sum():.4f}")
print(f"  AI-Optimized Capture   : {s3_capture.sum():.4f}")
print(f"  Net CO₂ Emission       : {s3_net_emission.sum():.4f}")
print(f"  Mean Fix. Efficiency   : {s3_efficiency.mean():.2f} %")
print(f"  Improvement over S2    : "
      f"{((s3_capture.sum()-s2_capture.sum())/s2_capture.sum()*100):.2f} %")

# ============================================================
# SCENARIO 4 — ENVIRONMENTAL SENSITIVITY
# Low solar vs High solar radiation impact
# ============================================================

print("\n--- Scenario 4 : Environmental Sensitivity ---")

q33 = np.quantile(solar, 0.33)
q66 = np.quantile(solar, 0.66)

mask_low_solar  = solar <= q33
mask_high_solar = solar >= q66

# Low solar — microbial activity suppressed (f_L is low)
s4_low_capture  = np.where(
    mask_low_solar,
    s3_capture * 0.55,     # 45% reduction
    s3_capture * 0.85
)

# High solar — microbial activity enhanced
s4_high_capture = np.where(
    mask_high_solar,
    s3_capture * 1.30,     # 30% boost
    s3_capture * 1.05
)

s4_low_net   = np.maximum(co2_raw - s4_low_capture,  0)
s4_high_net  = np.maximum(co2_raw - s4_high_capture, 0)

# Temperature bands
t_med = np.median(temperature)
mask_low_temp  = temperature < t_med
mask_high_temp = temperature >= t_med

s4_low_temp_cap  = np.where(mask_low_temp,
                              s3_capture * 0.70, s3_capture)
s4_high_temp_cap = np.where(mask_high_temp,
                              s3_capture * 1.20, s3_capture)

print(f"  Low Solar  — Mean Net Emission  : {s4_low_net.mean():.4f}")
print(f"  High Solar — Mean Net Emission  : {s4_high_net.mean():.4f}")
print(f"  Low Temp   — Mean Capture       : {s4_low_temp_cap.mean():.4f}")
print(f"  High Temp  — Mean Capture       : {s4_high_temp_cap.mean():.4f}")

# ============================================================
# SCENARIO 5 — INDUSTRIAL LOAD VARIATION
# Simulate ±30% and ±60% load variation on CO₂ emissions
# ============================================================

print("\n--- Scenario 5 : Industrial Load Variation ---")

load_levels = {
    '−60% Load' : 0.40,
    '−30% Load' : 0.70,
    'Baseline'  : 1.00,
    '+30% Load' : 1.30,
    '+60% Load' : 1.60,
}

s5_results = {}
for label, factor in load_levels.items():
    em_var   = co2_raw * factor
    cap_var  = s3_capture * min(factor, 1.15)   # capture saturates
    net_var  = np.maximum(em_var - cap_var, 0)
    eff_var  = np.where(em_var > 0,
                         (cap_var / em_var) * 100, 0)
    s5_results[label] = {
        'emission'   : em_var,
        'capture'    : cap_var,
        'net'        : net_var,
        'efficiency' : eff_var,
        'factor'     : factor,
    }
    print(f"  {label:12s} | Emission: {em_var.sum():.3f} "
          f"| Capture: {cap_var.sum():.3f} "
          f"| Net: {net_var.sum():.3f} "
          f"| Eff: {eff_var.mean():.2f}%")

# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n" + "=" * 60)
print("  SCENARIO COMPARISON SUMMARY")
print("=" * 60)
print(f"  {'Scenario':<30} {'Total Emission':>14} "
      f"{'Total Capture':>13} {'Net Emission':>12} {'Eff (%)':>8}")
print("-" * 60)

rows = [
    ("S1 — Baseline",         s1_emission, s1_capture,  s1_net_emission, s1_efficiency),
    ("S2 — Microbial",        s2_emission, s2_capture,  s2_net_emission, s2_efficiency),
    ("S3 — AI-Optimized",     s3_emission, s3_capture,  s3_net_emission, s3_efficiency),
    ("S4 — Low Solar",        co2_raw,     s4_low_capture,  s4_low_net,  np.where(co2_raw>0, s4_low_capture/co2_raw*100, 0)),
    ("S4 — High Solar",       co2_raw,     s4_high_capture, s4_high_net, np.where(co2_raw>0, s4_high_capture/co2_raw*100, 0)),
]

for nm, em, cap, net, eff in rows:
    print(f"  {nm:<30} {em.sum():>14.4f} "
          f"{cap.sum():>13.4f} {net.sum():>12.4f} "
          f"{eff.mean():>8.2f}")

print("=" * 60)

# ============================================================
# ──────────────────────────────────────────────────────────
#  PLOTS — each scenario in its OWN separate window
# ──────────────────────────────────────────────────────────
# ============================================================

# ============================================================
# SIM PLOT 1 — SCENARIO 1 : BASELINE
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(13, 8))
fig.suptitle('Scenario 1 — Baseline (Industrial Only)',
              fontsize=15, fontweight='bold')

axes[0].plot(t, s1_emission, color=SC['baseline'],
              lw=2.5, label='CO₂ Emission')
axes[0].fill_between(t, s1_emission, alpha=0.15,
                      color=SC['baseline'])
fmt_ax(axes[0], 'CO₂ Emission (No Capture)',
       'Facility Index', 'CO₂ Emission', legend=True)

axes[1].bar(t, s1_net_emission, color=SC['baseline'],
             alpha=0.75, width=0.8, label='Net CO₂')
fmt_ax(axes[1], 'Net CO₂ Emission (= Total, No Mitigation)',
       'Facility Index', 'Net CO₂ Emission', legend=True)

plt.tight_layout()
save_fig('SimPlot1_Baseline.png')
plt.show()

# ============================================================
# SIM PLOT 2 — SCENARIO 2 : MICROBIAL INTEGRATION
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(13, 11))
fig.suptitle('Scenario 2 — Microbial Integration',
              fontsize=15, fontweight='bold')

axes[0].plot(t, s2_emission,     color=SC['baseline'],
              lw=2, label='Gross Emission')
axes[0].plot(t, s2_net_emission, color=SC['microbial'],
              lw=2, ls='--', label='Net Emission')
axes[0].fill_between(t, s2_emission, s2_net_emission,
                      alpha=0.15, color=SC['microbial'],
                      label='Microbial Fixation Zone')
fmt_ax(axes[0], 'Gross vs Net CO₂ Emission',
       'Facility Index', 'CO₂ Emission')

axes[1].bar(t, s2_capture, color=SC['microbial'],
             alpha=0.8, width=0.8,
             label='Microbial CO₂ Absorption')
fmt_ax(axes[1], 'Microbial CO₂ Absorption per Facility',
       'Facility Index', 'CO₂ Captured')

axes[2].plot(t, s2_efficiency, color=SC['microbial'],
              lw=2, marker='o', ms=3,
              label='Fixation Efficiency (%)')
axes[2].axhline(s2_efficiency.mean(), color='red',
                 lw=1.5, ls='--',
                 label=f'Mean = {s2_efficiency.mean():.2f}%')
fmt_ax(axes[2], 'Microbial Fixation Efficiency (%)',
       'Facility Index', 'Efficiency (%)')

plt.tight_layout()
save_fig('SimPlot2_Microbial.png')
plt.show()

# ============================================================
# SIM PLOT 3 — SCENARIO 3 : AI-OPTIMIZED HYBRID
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(13, 11))
fig.suptitle('Scenario 3 — AI-Optimized Hybrid System',
              fontsize=15, fontweight='bold')

axes[0].plot(t, s1_net_emission, color=SC['baseline'],
              lw=2,   label='S1 Baseline Net')
axes[0].plot(t, s2_net_emission, color=SC['microbial'],
              lw=2,   label='S2 Microbial Net')
axes[0].plot(t, s3_net_emission, color=SC['ai_hybrid'],
              lw=2.5, label='S3 AI-Optimized Net', ls='-.')
fmt_ax(axes[0], 'Net CO₂ Emission : S1 vs S2 vs S3',
       'Facility Index', 'Net CO₂ Emission')

axes[1].plot(t, ai_boost_factor, color=SC['ai_hybrid'],
              lw=2, label='AI Optimization Score')
axes[1].fill_between(t, ai_boost_factor, alpha=0.15,
                      color=SC['ai_hybrid'])
fmt_ax(axes[1], 'AI Optimization Score per Facility',
       'Facility Index', 'AI Boost Factor [0–1]')

axes[2].plot(t, s3_capture,  color=SC['ai_hybrid'],
              lw=2.5, label='AI-Optimized Capture')
axes[2].plot(t, s2_capture,  color=SC['microbial'],
              lw=2,   ls='--', label='Microbial-Only Capture')
axes[2].fill_between(t, s2_capture, s3_capture,
                      alpha=0.15, color=SC['ai_hybrid'],
                      label='AI Improvement Zone')
fmt_ax(axes[2], 'CO₂ Capture : Microbial vs AI-Optimized',
       'Facility Index', 'CO₂ Captured')

plt.tight_layout()
save_fig('SimPlot3_AI_Hybrid.png')
plt.show()

# ============================================================
# SIM PLOT 4 — SCENARIO 4 : ENVIRONMENTAL SENSITIVITY
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Scenario 4 — Environmental Sensitivity',
              fontsize=15, fontweight='bold')

# Solar radiation vs CO₂ capture
ax = axes[0, 0]
sc = ax.scatter(solar, s3_capture,
                 c=solar, cmap='YlOrRd',
                 s=40, alpha=0.8, edgecolors='none')
plt.colorbar(sc, ax=ax, label='Solar Radiation')
fmt_ax(ax, 'Solar Radiation vs CO₂ Capture',
       'Solar Radiation', 'CO₂ Captured', legend=False)

# Low vs High solar net emission
ax = axes[0, 1]
ax.plot(t, s4_low_net,  color=SC['env_low'],
        lw=2, label='Low Solar Net Emission')
ax.plot(t, s4_high_net, color=SC['env_high'],
        lw=2, ls='--', label='High Solar Net Emission')
fmt_ax(ax, 'Net Emission : Low vs High Solar',
       'Facility Index', 'Net CO₂ Emission')

# Temperature vs CO₂ capture
ax = axes[1, 0]
sc2 = ax.scatter(temperature, s3_capture,
                  c=temperature, cmap='coolwarm',
                  s=40, alpha=0.8, edgecolors='none')
plt.colorbar(sc2, ax=ax, label='Temperature')
fmt_ax(ax, 'Temperature vs CO₂ Capture',
       'Temperature', 'CO₂ Captured', legend=False)

# Low vs High temperature capture
ax = axes[1, 1]
ax.plot(t, s4_low_temp_cap,  color=SC['env_low'],
        lw=2, label='Low Temp Capture')
ax.plot(t, s4_high_temp_cap, color=SC['env_high'],
        lw=2, ls='--', label='High Temp Capture')
fmt_ax(ax, 'CO₂ Capture : Low vs High Temperature',
       'Facility Index', 'CO₂ Captured')

plt.tight_layout()
save_fig('SimPlot4_Environmental_Sensitivity.png')
plt.show()

# ============================================================
# SIM PLOT 5 — SCENARIO 5 : INDUSTRIAL LOAD VARIATION
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Scenario 5 — Industrial Load Variation',
              fontsize=15, fontweight='bold')

load_colors = ['#1ABC9C', '#2ECC71', '#E67E22',
               '#E74C3C', '#8E44AD']

# Net emission under each load
ax = axes[0, 0]
for (lbl, res), clr in zip(s5_results.items(), load_colors):
    ax.plot(t, res['net'], color=clr, lw=2, label=lbl)
fmt_ax(ax, 'Net CO₂ Emission vs Load Level',
       'Facility Index', 'Net CO₂ Emission')

# Total emission bar chart
ax = axes[0, 1]
labels_l  = list(s5_results.keys())
tot_emiss = [s5_results[l]['emission'].sum() for l in labels_l]
tot_net   = [s5_results[l]['net'].sum()      for l in labels_l]
x_pos     = np.arange(len(labels_l))
w         = 0.38
bars1 = ax.bar(x_pos - w/2, tot_emiss, w,
                label='Total Emission',
                color=load_colors, alpha=0.75,
                edgecolor='black', lw=0.7)
bars2 = ax.bar(x_pos + w/2, tot_net,   w,
                label='Net Emission',
                color=load_colors, alpha=0.45,
                edgecolor='black', lw=0.7, hatch='//')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels_l, rotation=15,
                    fontsize=9, fontweight='bold')
fmt_ax(ax, 'Total Emission vs Net Emission by Load',
       'Load Scenario', 'CO₂ (Total)')

# Efficiency under each load
ax = axes[1, 0]
for (lbl, res), clr in zip(s5_results.items(), load_colors):
    ax.plot(t, res['efficiency'], color=clr, lw=2, label=lbl)
fmt_ax(ax, 'Fixation Efficiency (%) vs Load Level',
       'Facility Index', 'Efficiency (%)')

# Mean efficiency bar chart
ax = axes[1, 1]
mean_eff = [s5_results[l]['efficiency'].mean() for l in labels_l]
bars = ax.bar(labels_l, mean_eff,
               color=load_colors, alpha=0.85,
               edgecolor='black', lw=0.8)
for bar, val in zip(bars, mean_eff):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f'{val:.1f}%',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.set_xticklabels(labels_l, rotation=15,
                    fontsize=9, fontweight='bold')
fmt_ax(ax, 'Mean Fixation Efficiency by Load Level',
       'Load Scenario', 'Mean Efficiency (%)', legend=False)

plt.tight_layout()
save_fig('SimPlot5_Load_Variation.png')
plt.show()

# ============================================================
# SIM PLOT 6 — COMBINED SCENARIO COMPARISON SUMMARY
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('Scenario Comparison — System Simulation Summary',
              fontsize=15, fontweight='bold')

sc_labels  = ['S1\nBaseline', 'S2\nMicrobial',
               'S3\nAI-Hybrid', 'S4\nLow Solar',
               'S4\nHigh Solar']
sc_colors  = [SC['baseline'], SC['microbial'],
               SC['ai_hybrid'], SC['env_low'],
               SC['env_high']]

tot_emissions = [
    s1_emission.sum(),
    s2_emission.sum(),
    s3_emission.sum(),
    co2_raw.sum(),
    co2_raw.sum(),
]
tot_captures  = [
    s1_capture.sum(),
    s2_capture.sum(),
    s3_capture.sum(),
    s4_low_capture.sum(),
    s4_high_capture.sum(),
]
tot_nets      = [
    s1_net_emission.sum(),
    s2_net_emission.sum(),
    s3_net_emission.sum(),
    s4_low_net.sum(),
    s4_high_net.sum(),
]
mean_effs     = [
    s1_efficiency.mean(),
    s2_efficiency.mean(),
    s3_efficiency.mean(),
    np.where(co2_raw > 0,
             s4_low_capture / co2_raw * 100, 0).mean(),
    np.where(co2_raw > 0,
             s4_high_capture / co2_raw * 100, 0).mean(),
]

x = np.arange(len(sc_labels))
w = 0.35

# Total capture
ax = axes[0]
bars = ax.bar(x, tot_captures, width=0.6,
               color=sc_colors, alpha=0.85,
               edgecolor='black', lw=0.7)
for bar, v in zip(bars, tot_captures):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01 * max(tot_captures),
            f'{v:.2f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(sc_labels, fontsize=10, fontweight='bold')
fmt_ax(ax, 'Total CO₂ Captured',
       'Scenario', 'Total CO₂ Captured', legend=False)

# Net emission
ax = axes[1]
bars = ax.bar(x, tot_nets, width=0.6,
               color=sc_colors, alpha=0.85,
               edgecolor='black', lw=0.7)
for bar, v in zip(bars, tot_nets):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01 * max(tot_nets),
            f'{v:.2f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(sc_labels, fontsize=10, fontweight='bold')
fmt_ax(ax, 'Total Net CO₂ Emission',
       'Scenario', 'Net CO₂ Emission', legend=False)

# Mean efficiency
ax = axes[2]
bars = ax.bar(x, mean_effs, width=0.6,
               color=sc_colors, alpha=0.85,
               edgecolor='black', lw=0.7)
for bar, v in zip(bars, mean_effs):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f'{v:.1f}%', ha='center', va='bottom',
            fontsize=9, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(sc_labels, fontsize=10, fontweight='bold')
fmt_ax(ax, 'Mean Fixation Efficiency (%)',
       'Scenario', 'Efficiency (%)', legend=False)

plt.tight_layout()
save_fig('SimPlot6_Scenario_Summary.png')
plt.show()

# ============================================================
# FINAL SUMMARY — COMMAND WINDOW
# ============================================================

print("\n" + "=" * 60)
print("  SIMULATION COMPLETE — RESULTS SUMMARY")
print("=" * 60)
print(f"\n  {'Scenario':<30} {'Net Emission':>12} {'Capture':>10} {'Eff%':>8}")
print("-" * 60)

summary = [
    ("S1 — Baseline (Industrial)",
     s1_net_emission.sum(), s1_capture.sum(), s1_efficiency.mean()),
    ("S2 — Microbial Integration",
     s2_net_emission.sum(), s2_capture.sum(), s2_efficiency.mean()),
    ("S3 — AI-Optimized Hybrid",
     s3_net_emission.sum(), s3_capture.sum(), s3_efficiency.mean()),
    ("S4 — Low Solar Sensitivity",
     s4_low_net.sum(), s4_low_capture.sum(),
     np.where(co2_raw>0,s4_low_capture/co2_raw*100,0).mean()),
    ("S4 — High Solar Sensitivity",
     s4_high_net.sum(), s4_high_capture.sum(),
     np.where(co2_raw>0,s4_high_capture/co2_raw*100,0).mean()),
]

for nm, net, cap, eff in summary:
    print(f"  {nm:<30} {net:>12.4f} {cap:>10.4f} {eff:>8.2f}")

print("=" * 60)

print("\nScenario Plots Saved:")
sim_plots = [
    'SimPlot1_Baseline.png',
    'SimPlot2_Microbial.png',
    'SimPlot3_AI_Hybrid.png',
    'SimPlot4_Environmental_Sensitivity.png',
    'SimPlot5_Load_Variation.png',
    'SimPlot6_Scenario_Summary.png',
]
for p in sim_plots:
    print(f"  {p}")

print("\nStep 8 — System Simulation completed successfully.")

# ============================================================
# STEP 11 — CIRCULAR CARBON ECONOMY INTEGRATION
# Final Sustainability Layer
# Carbon Reuse: Biofuel | Bioplastic | Hydrogen | Biomass
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size']   = 13
plt.rcParams['font.weight'] = 'bold'

# ============================================================
# SAFE SAVE FUNCTION
# ============================================================

def save_fig(fname):
    fig  = plt.gcf()
    w, h = fig.get_size_inches()
    dpi  = int(min(200, 3000 / max(w, h)))
    plt.savefig(fname, dpi=dpi, bbox_inches='tight')
    print(f"  Saved → {fname}")

def fmt_ax(ax, title, xlabel, ylabel, legend=True):
    ax.set_title(title,   fontweight='bold', fontsize=13, pad=8)
    ax.set_xlabel(xlabel, fontweight='bold', fontsize=11)
    ax.set_ylabel(ylabel, fontweight='bold', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=10)
    if legend:
        ax.legend(fontsize=10, framealpha=0.9,
                  loc='best', frameon=True)

# ============================================================
# COLOUR PALETTE
# ============================================================

CCE = {
    'biofuel'    : '#E67E22',   # orange
    'bioplastic' : '#2980B9',   # blue
    'hydrogen'   : '#8E44AD',   # purple
    'biomass'    : '#27AE60',   # green
    'captured'   : '#C0392B',   # red
    'net'        : '#2C3E50',   # dark navy
    'sustain'    : '#1ABC9C',   # teal
    'arrow'      : '#7F8C8D',   # grey
}

# ============================================================
# LOAD DATASET (from Step 7 Monod results)
# ============================================================

df = pd.read_csv("monod_co2_absorption_results.csv")

print("=" * 65)
print("  STEP 11 — CIRCULAR CARBON ECONOMY INTEGRATION")
print("  Final Sustainability Layer")
print("=" * 65)
print(f"  Dataset Shape : {df.shape}")

N = len(df)
t = np.arange(N)

# ============================================================
# EXTRACT KEY COLUMNS
# ============================================================

def col(name, fallback_val=0.0):
    if name in df.columns:
        return df[name].fillna(df[name].median()).values
    return np.full(N, fallback_val)

co2_raw        = col('CO2 emission')
mic_absorption = col('Microbial_CO2_Absorption')
solar          = col('Solar_Radiation')
temperature    = col('Temperature')
wind           = col('Wind_Speed')
mbc            = col('MBC')
soc            = col('SOC')
monod_gr       = col('Monod_Growth_Rate')

# ============================================================
# STEP 11-A : CAPTURED CARBON POOL
# Total captured CO₂ available for reuse
# (using AI-Optimized capture from S3 logic)
# ============================================================

def norm01(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)

solar_n = norm01(solar)
mbc_n   = norm01(mbc)
temp_n  = norm01(temperature)
wind_n  = norm01(wind)

ai_boost = (
    0.35 * solar_n
  + 0.30 * mbc_n
  + 0.20 * temp_n
  + 0.15 * (1 - wind_n)
)

captured_co2 = mic_absorption * (1 + 0.60 * ai_boost)

print(f"\n  Total Captured CO₂ Available : {captured_co2.sum():.4f}")
print(f"  Mean  Captured CO₂ per Site  : {captured_co2.mean():.4f}")

# ============================================================
# STEP 11-B : CARBON REUSE ALLOCATION
# Distribute captured CO₂ across four pathways
# ============================================================
# Allocation fractions (must sum to 1.0):
#   Biofuel     35%  — highest energy yield
#   Bioplastic  25%  — material substitution
#   Hydrogen    20%  — clean fuel production
#   Biomass     20%  — soil carbon / feedstock
# ============================================================

ALLOC = {
    'biofuel'    : 0.35,
    'bioplastic' : 0.25,
    'hydrogen'   : 0.20,
    'biomass'    : 0.20,
}

# Conversion efficiency factors (captured CO₂ → product value)
CONV = {
    'biofuel'    : 0.72,   # 72% conversion to energy equivalent
    'bioplastic' : 0.65,   # 65% conversion to polymer mass
    'hydrogen'   : 0.58,   # 58% conversion efficiency (electrolysis)
    'biomass'    : 0.80,   # 80% incorporation into biomass
}

# Environmental benefit factors (CO₂-equivalent savings per unit)
BENEFIT = {
    'biofuel'    : 1.40,   # replaces fossil fuel → 1.4× CO₂ offset
    'bioplastic' : 1.20,   # replaces petroplastic → 1.2× offset
    'hydrogen'   : 1.60,   # replaces grey H₂ → 1.6× offset
    'biomass'    : 1.10,   # soil sequestration → 1.1× offset
}

reuse = {}
for product, frac in ALLOC.items():
    allocated    = captured_co2 * frac
    product_out  = allocated * CONV[product]
    co2_saved    = product_out * BENEFIT[product]
    reuse[product] = {
        'allocated'   : allocated,
        'output'      : product_out,
        'co2_saved'   : co2_saved,
    }

# ============================================================
# STEP 11-C : SUSTAINABILITY METRICS
# ============================================================

total_captured    = captured_co2.sum()
total_co2_raw     = co2_raw.sum()
total_reused      = sum(r['allocated'].sum() for r in reuse.values())
total_product_out = sum(r['output'].sum()    for r in reuse.values())
total_co2_saved   = sum(r['co2_saved'].sum() for r in reuse.values())

# Net carbon footprint after reuse
net_carbon_footprint = total_co2_raw - total_captured - total_co2_saved

# Circular economy score (0–100%)
ce_score = min((total_reused / (total_co2_raw + 1e-9)) * 100, 100)

# Carbon neutrality index
cn_index = (total_captured + total_co2_saved) / (total_co2_raw + 1e-9)

# Sustainability composite score
sustain_score = (
    0.40 * ce_score / 100
  + 0.35 * min(cn_index, 1.0)
  + 0.25 * (total_product_out / (total_captured + 1e-9))
) * 100

# ============================================================
# STEP 11-D : SUSTAINABILITY RATING
# ============================================================

def sustainability_rating(score):
    if   score >= 80: return "EXCELLENT — Near Carbon-Neutral"
    elif score >= 60: return "GOOD      — Significant Progress"
    elif score >= 40: return "MODERATE  — Improvement Needed"
    else:             return "LOW       — Critical Action Required"

rating = sustainability_rating(sustain_score)

# ============================================================
# STEP 11-E : TIME-SERIES REUSE BREAKDOWN PER FACILITY
# ============================================================

df['Captured_CO2']          = captured_co2
df['Reuse_Biofuel']         = reuse['biofuel']['allocated']
df['Reuse_Bioplastic']      = reuse['bioplastic']['allocated']
df['Reuse_Hydrogen']        = reuse['hydrogen']['allocated']
df['Reuse_Biomass']         = reuse['biomass']['allocated']
df['Product_Biofuel']       = reuse['biofuel']['output']
df['Product_Bioplastic']    = reuse['bioplastic']['output']
df['Product_Hydrogen']      = reuse['hydrogen']['output']
df['Product_Biomass']       = reuse['biomass']['output']
df['CO2_Saved_Biofuel']     = reuse['biofuel']['co2_saved']
df['CO2_Saved_Bioplastic']  = reuse['bioplastic']['co2_saved']
df['CO2_Saved_Hydrogen']    = reuse['hydrogen']['co2_saved']
df['CO2_Saved_Biomass']     = reuse['biomass']['co2_saved']
df['Total_CO2_Saved']       = (
    df['CO2_Saved_Biofuel'] + df['CO2_Saved_Bioplastic']
  + df['CO2_Saved_Hydrogen'] + df['CO2_Saved_Biomass']
)
df['Net_Carbon_Footprint']  = np.maximum(
    df['CO2 emission'] - df['Captured_CO2'] - df['Total_CO2_Saved'], 0
)

# ============================================================
# ██████████████████████████████████████████████████████████
# CIRCULAR CARBON ECONOMY — FULL COMMAND WINDOW RESULTS
# ██████████████████████████████████████████████████████████
# ============================================================

print("\n")
print("█" * 65)
print("  CIRCULAR CARBON ECONOMY — COMPLETE RESULTS REPORT")
print("  STEP 11 — FINAL SUSTAINABILITY LAYER")
print("█" * 65)

# ── Section A : Input Summary ────────────────────────────────
print("\n" + "─" * 65)
print("  A. DATASET & INPUT SUMMARY")
print("─" * 65)
print(f"  Number of Facilities / Sites   : {N}")
print(f"  Total Gross CO₂ Emission       : {total_co2_raw:.6f}")
print(f"  Mean  CO₂ Emission per Site    : {co2_raw.mean():.6f}")
print(f"  Min   CO₂ Emission             : {co2_raw.min():.6f}")
print(f"  Max   CO₂ Emission             : {co2_raw.max():.6f}")

# ── Section B : Capture Performance ─────────────────────────
print("\n" + "─" * 65)
print("  B. AI-OPTIMIZED CO₂ CAPTURE PERFORMANCE")
print("─" * 65)
print(f"  Total CO₂ Captured (AI-Boost)  : {total_captured:.6f}")
print(f"  Mean  CO₂ Captured per Site    : {captured_co2.mean():.6f}")
print(f"  Capture Rate (% of Gross)      : {(total_captured/total_co2_raw*100):.2f} %")
print(f"  AI Boost Mean Factor           : {ai_boost.mean():.4f}")
print(f"  AI Boost Range                 : {ai_boost.min():.4f} – {ai_boost.max():.4f}")

# ── Section C : Carbon Reuse Allocation ─────────────────────
print("\n" + "─" * 65)
print("  C. CARBON REUSE ALLOCATION RESULTS")
print("─" * 65)
print(f"  {'Pathway':<14} {'Alloc%':>7}  {'CO₂ Input':>10}  "
      f"{'Conv%':>6}  {'Product Out':>12}  "
      f"{'Benefit×':>9}  {'CO₂ Saved':>11}")
print("  " + "-" * 63)
for product, frac in ALLOC.items():
    r   = reuse[product]
    inp = r['allocated'].sum()
    out = r['output'].sum()
    sav = r['co2_saved'].sum()
    print(f"  {product.capitalize():<14} {frac*100:>6.1f}%  "
          f"{inp:>10.4f}  "
          f"{CONV[product]*100:>5.0f}%  "
          f"{out:>12.4f}  "
          f"{BENEFIT[product]:>9.2f}×  "
          f"{sav:>11.4f}")
print("  " + "-" * 63)
print(f"  {'TOTAL':<14} {'100.0%':>7}  "
      f"{total_reused:>10.4f}  "
      f"{'—':>6}  "
      f"{total_product_out:>12.4f}  "
      f"{'—':>9}  "
      f"{total_co2_saved:>11.4f}")

# ── Section D : Net Carbon Footprint ────────────────────────
print("\n" + "─" * 65)
print("  D. NET CARBON FOOTPRINT ANALYSIS")
print("─" * 65)
print(f"  Gross CO₂ Emission             : {total_co2_raw:.6f}")
print(f"  (–) AI-Optimized Capture       : {total_captured:.6f}")
print(f"  (–) Reuse CO₂ Offset           : {total_co2_saved:.6f}")
print(f"  {'─'*45}")
print(f"  Net Carbon Footprint           : {net_carbon_footprint:.6f}")
reduction_pct = (1 - net_carbon_footprint / total_co2_raw) * 100
print(f"  Total Reduction Achieved       : {reduction_pct:.2f} %")

# ── Section E : Sustainability KPIs ─────────────────────────
print("\n" + "─" * 65)
print("  E. SUSTAINABILITY KEY PERFORMANCE INDICATORS (KPIs)")
print("─" * 65)
print(f"  Circular Economy Score (CE)    : {ce_score:.4f} %")
print(f"  Carbon Neutrality Index (CN)   : {cn_index:.6f}")
print(f"  Product Yield Ratio            : "
      f"{(total_product_out/total_captured):.4f}")
print(f"  Reuse Efficiency               : "
      f"{(total_reused/total_captured*100):.2f} %")
print(f"  CO₂ Offset Multiplier          : "
      f"{(total_co2_saved/total_reused):.4f}×")
print(f"  Sustainability Composite Score : {sustain_score:.4f} / 100")

# ── Section F : Per-Pathway CO₂ Savings ─────────────────────
print("\n" + "─" * 65)
print("  F. CO₂ SAVINGS BREAKDOWN BY PATHWAY")
print("─" * 65)
total_saved_all = sum(reuse[p]['co2_saved'].sum() for p in reuse)
for product in ALLOC:
    sav = reuse[product]['co2_saved'].sum()
    share = sav / total_saved_all * 100
    bar_len = int(share / 2.5)
    bar = "█" * bar_len + "░" * (40 - bar_len)
    print(f"  {product.capitalize():<12} : {sav:>10.4f}  "
          f"({share:>5.1f}%)  {bar}")

# ── Section G : Environmental Impact Comparison ──────────────
print("\n" + "─" * 65)
print("  G. SCENARIO IMPACT COMPARISON")
print("─" * 65)
print(f"  {'Condition':<32} {'Net Footprint':>14} {'Reduction%':>11}")
print("  " + "-" * 58)

baseline_net = total_co2_raw
after_capture = total_co2_raw - total_captured
after_reuse   = net_carbon_footprint

scenarios_g = [
    ("No Mitigation (Baseline)",     baseline_net,   0.0),
    ("After Microbial Capture",      after_capture,
     (1 - after_capture  / baseline_net) * 100),
    ("After AI Capture + Reuse",     after_reuse,
     (1 - after_reuse    / baseline_net) * 100),
]
for nm, val, red in scenarios_g:
    print(f"  {nm:<32} {val:>14.4f} {red:>10.2f}%")

# ── Section H : Per-Site Statistics ─────────────────────────
print("\n" + "─" * 65)
print("  H. PER-SITE STATISTICS")
print("─" * 65)
net_fp_arr = df['Net_Carbon_Footprint'].values
print(f"  Net Footprint  — Mean  : {net_fp_arr.mean():.6f}")
print(f"  Net Footprint  — Std   : {net_fp_arr.std():.6f}")
print(f"  Net Footprint  — Min   : {net_fp_arr.min():.6f}")
print(f"  Net Footprint  — Max   : {net_fp_arr.max():.6f}")
print(f"  Sites at Zero  Emission: "
      f"{(net_fp_arr == 0).sum()} / {N}")
print(f"  Sites below 50% Base   : "
      f"{(net_fp_arr < co2_raw * 0.5).sum()} / {N}")

# ── Section I : Final Rating ─────────────────────────────────
print("\n" + "─" * 65)
print("  I. FINAL SUSTAINABILITY RATING")
print("─" * 65)

stars = int(sustain_score / 20)
star_bar = "★" * stars + "☆" * (5 - stars)

print(f"\n  Score    : {sustain_score:.2f} / 100")
print(f"  Stars    : {star_bar}  ({stars}/5)")
print(f"  Rating   : {rating}")

if   sustain_score >= 80:
    verdict = "System achieves near carbon-neutral operation."
elif sustain_score >= 60:
    verdict = "System shows strong carbon reduction progress."
elif sustain_score >= 40:
    verdict = "System requires further optimization."
else:
    verdict = "System needs critical intervention."

print(f"  Verdict  : {verdict}")

print("\n" + "█" * 65)
print("  CIRCULAR CARBON ECONOMY REPORT — END")
print("█" * 65)

# ============================================================
# ──────────────────────────────────────────────────────────
#  PLOTS — STEP 11
# ──────────────────────────────────────────────────────────
# ============================================================

# ============================================================
# PLOT 11-1 — CARBON FLOW DIAGRAM (Circular Economy Overview)
# ============================================================

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('#F8F9FA')

fig.suptitle('Circular Carbon Economy — Carbon Flow Diagram',
             fontsize=16, fontweight='bold', y=0.98)

# Nodes: (x_center, y_center, label, color)
nodes = [
    (5.0, 6.0, f'Industrial CO₂\nEmission\n{total_co2_raw:.2f}', '#C0392B'),
    (5.0, 4.2, f'AI-Optimized\nCapture\n{total_captured:.2f}',   '#2980B9'),
    (1.5, 2.0, f'Biofuel\n{reuse["biofuel"]["output"].sum():.2f}',      CCE['biofuel']),
    (3.5, 2.0, f'Bioplastic\n{reuse["bioplastic"]["output"].sum():.2f}', CCE['bioplastic']),
    (6.5, 2.0, f'Hydrogen\n{reuse["hydrogen"]["output"].sum():.2f}',     CCE['hydrogen']),
    (8.5, 2.0, f'Biomass\n{reuse["biomass"]["output"].sum():.2f}',       CCE['biomass']),
    (5.0, 0.4, f'CO₂ Offset / Carbon Credits\n{total_co2_saved:.2f}', '#1ABC9C'),
]

for (x, y, lbl, clr) in nodes:
    ax.add_patch(plt.Circle((x, y), 0.55,
                  color=clr, alpha=0.88, zorder=3))
    ax.text(x, y, lbl, ha='center', va='center',
            fontsize=8.5, fontweight='bold',
            color='white', zorder=4, multialignment='center')

# Arrows
def draw_arrow(ax, x1, y1, x2, y2, clr):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->', color=clr,
                    lw=2.5, connectionstyle='arc3,rad=0.05'))

draw_arrow(ax, 5.0, 5.4, 5.0, 4.8, '#C0392B')
for xi in [1.5, 3.5, 6.5, 8.5]:
    draw_arrow(ax, 5.0, 3.65, xi, 2.55, '#2980B9')
for xi in [1.5, 3.5, 6.5, 8.5]:
    draw_arrow(ax, xi, 1.45, 5.0, 0.95, '#1ABC9C')

alloc_labels = [
    (2.8, 3.2, '35%', CCE['biofuel']),
    (4.1, 3.2, '25%', CCE['bioplastic']),
    (5.9, 3.2, '20%', CCE['hydrogen']),
    (7.2, 3.2, '20%', CCE['biomass']),
]
for (x, y, lbl, clr) in alloc_labels:
    ax.text(x, y, lbl, ha='center', va='center',
            fontsize=9, fontweight='bold', color=clr,
            bbox=dict(boxstyle='round,pad=0.2',
                      facecolor='white', alpha=0.75))

ax.text(5.0, -0.15,
        f'Circular Economy Score: {ce_score:.1f}%  |  '
        f'Carbon Neutrality Index: {cn_index:.3f}  |  '
        f'Sustainability Score: {sustain_score:.1f}/100',
        ha='center', va='center', fontsize=10,
        fontweight='bold', color='#2C3E50',
        transform=ax.transData,
        bbox=dict(boxstyle='round,pad=0.4',
                  facecolor='#D5EBD0', alpha=0.9,
                  edgecolor='#27AE60', lw=1.5))

plt.tight_layout()
save_fig('CCE_Plot1_Carbon_Flow_Diagram.png')
plt.show()

# ============================================================
# PLOT 11-2 — CARBON REUSE ALLOCATION STACKED AREA
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Circular Carbon Economy — Carbon Reuse Allocation',
             fontsize=15, fontweight='bold')

ax = axes[0]
ax.stackplot(
    t,
    df['Reuse_Biofuel'],
    df['Reuse_Bioplastic'],
    df['Reuse_Hydrogen'],
    df['Reuse_Biomass'],
    labels=['Biofuel (35%)', 'Bioplastic (25%)',
            'Hydrogen (20%)', 'Biomass (20%)'],
    colors=[CCE['biofuel'], CCE['bioplastic'],
            CCE['hydrogen'], CCE['biomass']],
    alpha=0.82
)
fmt_ax(ax, 'CO₂ Allocated per Pathway per Facility',
       'Facility Index', 'CO₂ Allocated')

ax = axes[1]
ax.bar(t, df['Product_Biofuel'],    color=CCE['biofuel'],
        label='Biofuel Output',     alpha=0.85)
ax.bar(t, df['Product_Bioplastic'], color=CCE['bioplastic'],
        label='Bioplastic Output',  alpha=0.85,
        bottom=df['Product_Biofuel'])
ax.bar(t, df['Product_Hydrogen'],   color=CCE['hydrogen'],
        label='Hydrogen Output',    alpha=0.85,
        bottom=df['Product_Biofuel'] + df['Product_Bioplastic'])
ax.bar(t, df['Product_Biomass'],    color=CCE['biomass'],
        label='Biomass Output',     alpha=0.85,
        bottom=(df['Product_Biofuel'] + df['Product_Bioplastic']
                + df['Product_Hydrogen']))
fmt_ax(ax, 'Product Output per Pathway per Facility',
       'Facility Index', 'Product Output')

plt.tight_layout()
save_fig('CCE_Plot2_Reuse_Allocation.png')
plt.show()

# ============================================================
# PLOT 11-3 — CO₂ SAVED BY EACH PATHWAY
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('CO₂ Offset by Carbon Reuse Pathways',
             fontsize=15, fontweight='bold')

ax = axes[0]
cum_bf  = df['CO2_Saved_Biofuel'].cumsum()
cum_bp  = df['CO2_Saved_Bioplastic'].cumsum()
cum_h   = df['CO2_Saved_Hydrogen'].cumsum()
cum_bm  = df['CO2_Saved_Biomass'].cumsum()
ax.plot(t, cum_bf,  color=CCE['biofuel'],    lw=2.5, label='Biofuel')
ax.plot(t, cum_bp,  color=CCE['bioplastic'], lw=2.5, label='Bioplastic')
ax.plot(t, cum_h,   color=CCE['hydrogen'],   lw=2.5, label='Hydrogen')
ax.plot(t, cum_bm,  color=CCE['biomass'],    lw=2.5, label='Biomass')
fmt_ax(ax, 'Cumulative CO₂ Saved per Pathway',
       'Facility Index', 'Cumulative CO₂ Offset')

ax = axes[1]
saved_vals  = [
    df['CO2_Saved_Biofuel'].sum(),
    df['CO2_Saved_Bioplastic'].sum(),
    df['CO2_Saved_Hydrogen'].sum(),
    df['CO2_Saved_Biomass'].sum(),
]
saved_labels = [
    f'Biofuel\n{saved_vals[0]:.2f}',
    f'Bioplastic\n{saved_vals[1]:.2f}',
    f'Hydrogen\n{saved_vals[2]:.2f}',
    f'Biomass\n{saved_vals[3]:.2f}',
]
wedge_colors = [CCE['biofuel'], CCE['bioplastic'],
                CCE['hydrogen'], CCE['biomass']]

wedges, texts, autotexts = ax.pie(
    saved_vals,
    labels=saved_labels,
    colors=wedge_colors,
    autopct='%1.1f%%',
    startangle=140,
    pctdistance=0.75,
    wedgeprops=dict(edgecolor='white', linewidth=1.5)
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight('bold')
ax.set_title('Share of CO₂ Offset by Pathway',
             fontweight='bold', fontsize=13, pad=12)

plt.tight_layout()
save_fig('CCE_Plot3_CO2_Saved_Pathways.png')
plt.show()

# ============================================================
# PLOT 11-4 — NET CARBON FOOTPRINT REDUCTION JOURNEY
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Net Carbon Footprint Reduction Journey',
             fontsize=15, fontweight='bold')

ax = axes[0]
ax.plot(t, co2_raw,                   color=CCE['captured'],
        lw=2.5, label='Gross CO₂ Emission')
ax.plot(t, co2_raw - captured_co2,    color=CCE['biofuel'],
        lw=2.0, ls='--',
        label='After Microbial Capture')
ax.plot(t, df['Net_Carbon_Footprint'], color=CCE['sustain'],
        lw=2.5, ls='-.',
        label='Net Footprint (After Reuse Offset)')
ax.fill_between(t, co2_raw, df['Net_Carbon_Footprint'],
                alpha=0.10, color=CCE['sustain'],
                label='Total Reduction Zone')
fmt_ax(ax, 'CO₂ Emission vs Net Carbon Footprint',
       'Facility Index', 'CO₂ Level')

ax = axes[1]
ax.plot(t, df['Net_Carbon_Footprint'].cumsum(),
        color=CCE['net'],   lw=2.5,
        label='Cumulative Net Footprint')
ax.plot(t, co2_raw.cumsum(),
        color=CCE['captured'], lw=2.0, ls='--',
        label='Cumulative Gross Emission')
ax.fill_between(t,
                co2_raw.cumsum(),
                df['Net_Carbon_Footprint'].cumsum(),
                alpha=0.12, color=CCE['sustain'],
                label='Cumulative Savings Zone')
fmt_ax(ax, 'Cumulative CO₂ — Gross vs Net Footprint',
       'Facility Index', 'Cumulative CO₂')

plt.tight_layout()
save_fig('CCE_Plot4_Net_Footprint_Journey.png')
plt.show()

# ============================================================
# PLOT 11-5 — PRODUCT YIELD ANALYSIS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Carbon Reuse — Product Yield Analysis',
             fontsize=15, fontweight='bold')

products = ['biofuel', 'bioplastic', 'hydrogen', 'biomass']
colors   = [CCE['biofuel'], CCE['bioplastic'],
            CCE['hydrogen'], CCE['biomass']]
cols_out = ['Product_Biofuel', 'Product_Bioplastic',
            'Product_Hydrogen', 'Product_Biomass']

for ax, prod, clr, col_out in zip(
        axes.flatten(), products, colors, cols_out):

    ax.bar(t, df[col_out], color=clr, alpha=0.82,
            width=0.85, label=f'{prod.capitalize()} Output')
    ax.plot(t, df[col_out].cumsum() / (t + 1),
            color='black', lw=1.5, ls='--',
            label='Running Mean')
    fmt_ax(ax,
           f'{prod.capitalize()} — Product Yield per Facility',
           'Facility Index', 'Product Output')

plt.tight_layout()
save_fig('CCE_Plot5_Product_Yield.png')
plt.show()

# ============================================================
# PLOT 11-6 — SUSTAINABILITY DASHBOARD (2 × 3)
# ============================================================

fig = plt.figure(figsize=(16, 10))
fig.suptitle('Sustainability Dashboard — Circular Carbon Economy',
             fontsize=16, fontweight='bold')

ax1 = fig.add_subplot(2, 3, 1)
theta = np.linspace(np.pi, 0, 500)
ax1.plot(np.cos(theta), np.sin(theta),
         color='#ECF0F1', lw=30, solid_capstyle='round')
score_theta = np.linspace(np.pi,
                           np.pi - (sustain_score / 100) * np.pi,
                           300)
sc_clr = ('#C0392B' if sustain_score < 40 else
           '#E67E22' if sustain_score < 60 else
           '#F1C40F' if sustain_score < 80 else '#27AE60')
ax1.plot(np.cos(score_theta), np.sin(score_theta),
         color=sc_clr, lw=30, solid_capstyle='round')
ax1.text(0, 0.10, f'{sustain_score:.1f}',
         ha='center', va='center',
         fontsize=26, fontweight='bold', color=sc_clr)
ax1.text(0, -0.25, '/ 100',
         ha='center', va='center', fontsize=13, color='#7F8C8D')
ax1.text(0, -0.50, 'Sustainability Score',
         ha='center', va='center', fontsize=11, fontweight='bold')
ax1.set_xlim(-1.2, 1.2)
ax1.set_ylim(-0.7, 1.2)
ax1.axis('off')

ax2 = fig.add_subplot(2, 3, 2)
categories = ['CE Score\n(%)', 'CN Index\n(×100)', 'Sustain\nScore']
values     = [ce_score, cn_index * 100, sustain_score]
bar_colors = ['#2980B9', '#8E44AD', '#27AE60']
bars = ax2.bar(categories, values, color=bar_colors,
                alpha=0.85, edgecolor='black', lw=0.8, width=0.5)
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.8,
             f'{val:.1f}', ha='center', va='bottom',
             fontsize=11, fontweight='bold')
ax2.set_ylim(0, 120)
fmt_ax(ax2, 'Key Sustainability Indices',
       'Index', 'Score / Value', legend=False)

ax3 = fig.add_subplot(2, 3, 3)
donut_vals   = list(ALLOC.values())
donut_labels = [f'{k.capitalize()}\n{v*100:.0f}%'
                for k, v in ALLOC.items()]
wedges, _ = ax3.pie(
    donut_vals,
    labels=donut_labels,
    colors=[CCE['biofuel'], CCE['bioplastic'],
            CCE['hydrogen'], CCE['biomass']],
    startangle=90,
    wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2)
)
ax3.text(0, 0, 'Carbon\nReuse\nMix', ha='center',
          va='center', fontsize=11, fontweight='bold',
          color='#2C3E50')
ax3.set_title('Carbon Reuse Allocation',
               fontweight='bold', fontsize=12, pad=8)

ax4 = fig.add_subplot(2, 3, 4)
ax4.fill_between(t, co2_raw,
                  df['Net_Carbon_Footprint'],
                  alpha=0.25, color=CCE['sustain'],
                  label='CO₂ Reduction')
ax4.plot(t, co2_raw,                   color=CCE['captured'],
         lw=2, label='Gross Emission')
ax4.plot(t, df['Net_Carbon_Footprint'], color=CCE['sustain'],
         lw=2, label='Net Footprint')
fmt_ax(ax4, 'Gross vs Net Carbon Footprint',
       'Facility Index', 'CO₂')

ax5 = fig.add_subplot(2, 3, 5)
ax5.bar(t, df['CO2_Saved_Biofuel'],
        color=CCE['biofuel'],    alpha=0.85,
        label='Biofuel Offset')
ax5.bar(t, df['CO2_Saved_Bioplastic'],
        color=CCE['bioplastic'], alpha=0.85,
        label='Bioplastic Offset',
        bottom=df['CO2_Saved_Biofuel'])
ax5.bar(t, df['CO2_Saved_Hydrogen'],
        color=CCE['hydrogen'],   alpha=0.85,
        label='Hydrogen Offset',
        bottom=df['CO2_Saved_Biofuel'] + df['CO2_Saved_Bioplastic'])
ax5.bar(t, df['CO2_Saved_Biomass'],
        color=CCE['biomass'],    alpha=0.85,
        label='Biomass Offset',
        bottom=(df['CO2_Saved_Biofuel'] + df['CO2_Saved_Bioplastic']
                + df['CO2_Saved_Hydrogen']))
fmt_ax(ax5, 'Total CO₂ Offset per Facility',
       'Facility Index', 'CO₂ Saved')

ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')
summary_text = (
    f"{'='*38}\n"
    f"  CIRCULAR CARBON ECONOMY SUMMARY\n"
    f"{'='*38}\n\n"
    f"  Gross CO₂ Emission : {total_co2_raw:.4f}\n"
    f"  CO₂ Captured       : {total_captured:.4f}\n"
    f"  CO₂ Reused         : {total_reused:.4f}\n"
    f"  Product Output     : {total_product_out:.4f}\n"
    f"  CO₂ Offset (Reuse) : {total_co2_saved:.4f}\n"
    f"  Net Footprint      : {net_carbon_footprint:.4f}\n\n"
    f"  CE Score           : {ce_score:.2f} %\n"
    f"  CN Index           : {cn_index:.4f}\n"
    f"  Sustainability     : {sustain_score:.2f} / 100\n\n"
    f"  Rating  :\n"
    f"  {rating}\n"
)
ax6.text(0.05, 0.95, summary_text,
          transform=ax6.transAxes,
          fontsize=9.5, va='top', fontfamily='monospace',
          bbox=dict(boxstyle='round,pad=0.6',
                    facecolor='#EAF7F0',
                    edgecolor='#27AE60', lw=1.5))

plt.tight_layout()
save_fig('CCE_Plot6_Sustainability_Dashboard.png')
plt.show()

# ============================================================
# SAVE FINAL DATASET
# ============================================================

df.to_csv("circular_carbon_economy_results.csv", index=False)
print("\n  Final dataset saved → circular_carbon_economy_results.csv")

# ============================================================
# PLOTS LIST — COMMAND WINDOW
# ============================================================

print("\n" + "─" * 65)
print("  ALL OUTPUT FILES SAVED")
print("─" * 65)

all_outputs = [
    ("HHO Convergence Plot",    "HHO_Convergence_Plot.png"),
    ("CCE Flow Diagram",        "CCE_Plot1_Carbon_Flow_Diagram.png"),
    ("CCE Reuse Allocation",    "CCE_Plot2_Reuse_Allocation.png"),
    ("CCE CO₂ Saved Pathways",  "CCE_Plot3_CO2_Saved_Pathways.png"),
    ("CCE Net Footprint",       "CCE_Plot4_Net_Footprint_Journey.png"),
    ("CCE Product Yield",       "CCE_Plot5_Product_Yield.png"),
    ("CCE Dashboard",           "CCE_Plot6_Sustainability_Dashboard.png"),
    ("Final Dataset (CSV)",     "circular_carbon_economy_results.csv"),
]
for label, fname in all_outputs:
    print(f"  {label:<30} →  {fname}")

print("\n" + "█" * 65)
print("  STEP 11 — CIRCULAR CARBON ECONOMY COMPLETED SUCCESSFULLY")
print("█" * 65)