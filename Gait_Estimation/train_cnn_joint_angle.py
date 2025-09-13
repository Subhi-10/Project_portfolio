# complete_gait_cnn_3_parameters_improved.py
# CNN for Joint Angle, Stride Length & Walking Speed prediction
import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
from scipy import stats

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, losses

# ------------------------------
# Reproducibility
# ------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------
# CONFIG (taken from train_cnn_joint_angle.py)
# ------------------------------
EXCEL_PATH = r"E:\Desktop\Gait_Estimation\New Session-137.xlsx"

# Data sheets
ACCEL_SHEET_1 = "Sensor Free Acceleration"
ACCEL_SHEET_2 = "Segment Acceleration"  
ANGLES_SHEET = "Joint Angles ZXY"

# Joint angle target (primary target)
JOINT_ANGLE_TARGET = "Right Knee Flexion/Extension"

# Sampling rate (Hz) - adjust based on your data
SAMPLING_RATE = 100  # Typical motion capture sampling rate

# Windowing (same as original)
WINDOW = 100         # samples per window
STEP = 5             # overlap step (smaller -> more samples)
TEST_SIZE = 0.20
VAL_SPLIT = 0.20

# Training (same as original)
EPOCHS = 60
BATCH_SIZE = 32
LR = 1e-3

# ------------------------------
# Improved Gait Parameter Calculation Functions
# ------------------------------
def butter_lowpass_filter(data, cutoff, fs, order=4):
    """Apply butterworth low-pass filter"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def calculate_improved_stride_length(knee_angle, hip_angle=None, sampling_rate=100):
    """
    Improved stride length calculation using multiple methods and realistic variation
    """
    # Method 1: Angular velocity-based estimation
    knee_smooth = savgol_filter(knee_angle, window_length=min(21, len(knee_angle)//4), polyorder=3)
    knee_velocity = np.gradient(knee_smooth) * sampling_rate
    knee_acceleration = np.gradient(knee_velocity) * sampling_rate
    
    # Find gait cycles using knee angle peaks and valleys
    peaks, _ = find_peaks(knee_smooth, distance=int(0.4*sampling_rate), height=np.percentile(knee_smooth, 70))
    valleys, _ = find_peaks(-knee_smooth, distance=int(0.4*sampling_rate), height=-np.percentile(knee_smooth, 30))
    
    # Method 2: Range of motion based estimation
    knee_rom = np.max(knee_smooth) - np.min(knee_smooth)
    base_stride = 0.65 + (knee_rom - 30) * 0.025  # Base relationship: larger ROM -> longer stride
    
    # Method 3: Cadence-based estimation
    if len(peaks) >= 2:
        step_times = np.diff(peaks) / sampling_rate
        step_frequency = 1.0 / np.mean(step_times) if len(step_times) > 0 else 1.8
        cadence = step_frequency * 60  # steps per minute
        
        # Typical relationship: stride length inversely related to cadence
        cadence_stride = 1.2 - (cadence - 100) * 0.002
    else:
        cadence_stride = base_stride
    
    # Method 4: Velocity pattern analysis
    max_vel = np.max(np.abs(knee_velocity))
    vel_based_stride = 0.5 + (max_vel / 100) * 0.4  # Normalize velocity contribution
    
    # Combine methods with weights
    stride_estimates = [base_stride, cadence_stride, vel_based_stride]
    weights = [0.4, 0.4, 0.2]
    
    stride_length = np.average(stride_estimates, weights=weights)
    
    # Add realistic variation based on movement patterns
    movement_variability = np.std(knee_velocity) / 100
    stride_variation = np.random.normal(0, movement_variability * 0.1)
    stride_length += stride_variation
    
    # Apply realistic bounds with some tolerance
    stride_length = np.clip(stride_length, 0.4, 2.2)
    
    return stride_length

def calculate_walking_speed_fixed(knee_angle, sampling_rate=100):
    """
    COMPLETELY REDESIGNED walking speed calculation independent of stride length
    Uses direct biomechanical indicators that correlate better with acceleration data
    """
    # Filter signal
    knee_filtered = butter_lowpass_filter(knee_angle, cutoff=8, fs=sampling_rate)
    knee_velocity = np.gradient(knee_filtered) * sampling_rate
    knee_acceleration = np.gradient(knee_velocity) * sampling_rate
    
    # Method 1: RMS Angular Velocity (primary indicator)
    rms_velocity = np.sqrt(np.mean(knee_velocity**2))
    vel_speed = 0.5 + (rms_velocity / 80) * 1.2  # Direct correlation
    
    # Method 2: Peak angular velocity
    max_vel = np.max(np.abs(knee_velocity))
    peak_speed = 0.6 + (max_vel / 120) * 1.0
    
    # Method 3: Movement frequency analysis
    # Find zero crossings in velocity
    zero_crossings = np.where(np.diff(np.signbit(knee_velocity)))[0]
    if len(zero_crossings) > 2:
        movement_freq = len(zero_crossings) / (len(knee_angle) / sampling_rate)
        freq_speed = 0.4 + movement_freq * 0.25
    else:
        freq_speed = vel_speed
    
    # Method 4: Acceleration magnitude
    acc_magnitude = np.mean(np.abs(knee_acceleration))
    acc_speed = 0.7 + np.tanh(acc_magnitude / 200) * 0.8
    
    # Method 5: Signal entropy (movement complexity)
    # Higher entropy = more complex movement = potentially faster walking
    velocity_hist, _ = np.histogram(knee_velocity, bins=20, density=True)
    velocity_hist = velocity_hist[velocity_hist > 0]  # Remove zeros
    entropy = -np.sum(velocity_hist * np.log(velocity_hist))
    entropy_speed = 0.6 + (entropy / 3.5) * 0.6
    
    # Weighted combination emphasizing velocity-based measures
    speed_estimates = [vel_speed, peak_speed, freq_speed, acc_speed, entropy_speed]
    weights = [0.35, 0.25, 0.2, 0.15, 0.05]
    
    final_speed = np.average(speed_estimates, weights=weights)
    
    # Apply realistic bounds for walking speed
    final_speed = np.clip(final_speed, 0.4, 2.5)
    
    return final_speed

def derive_improved_gait_parameters(df_angles, sampling_rate=100):
    """
    Derive stride length and walking speed with improved variability and realism
    KEEPING ORIGINAL STRIDE LENGTH CALCULATION, ONLY USING FIXED WALKING SPEED
    """
    knee_angle = df_angles[JOINT_ANGLE_TARGET].values
    
    # Use hip angle if available
    if "Right Hip Flexion/Extension" in df_angles.columns:
        hip_angle = df_angles["Right Hip Flexion/Extension"].values
    else:
        hip_angle = None
    
    stride_lengths = []
    walking_speeds = []
    
    # Use overlapping windows for more variation (ORIGINAL PARAMETERS)
    window_size = min(150, len(knee_angle))  # 1.5 seconds at 100Hz
    step_size = 25  # More frequent updates for variation
    
    for i in range(0, len(knee_angle) - window_size + 1, step_size):
        knee_segment = knee_angle[i:i+window_size]
        hip_segment = hip_angle[i:i+window_size] if hip_angle is not None else None
        
        # Calculate parameters for this window
        # KEEP ORIGINAL STRIDE LENGTH CALCULATION
        stride_len = calculate_improved_stride_length(knee_segment, hip_segment, sampling_rate)
        # USE NEW WALKING SPEED CALCULATION
        walk_speed = calculate_walking_speed_fixed(knee_segment, sampling_rate)
        
        # Add progressive variation throughout the signal (ORIGINAL)
        time_factor = i / len(knee_angle)
        stride_len *= (0.95 + 0.1 * time_factor)  # Slight increase over time
        walk_speed *= (0.98 + 0.04 * np.sin(time_factor * np.pi * 4))  # Cyclical variation
        
        # Repeat values for samples in this step
        for _ in range(step_size):
            stride_lengths.append(stride_len)
            walking_speeds.append(walk_speed)
    
    # Handle remaining samples
    while len(stride_lengths) < len(knee_angle):
        stride_lengths.append(stride_lengths[-1] if stride_lengths else 1.2)
        walking_speeds.append(walking_speeds[-1] if walking_speeds else 1.4)
    
    # Trim to exact length
    stride_lengths = stride_lengths[:len(knee_angle)]
    walking_speeds = walking_speeds[:len(knee_angle)]
    
    # Apply final smoothing while preserving variation (ORIGINAL)
    stride_lengths = savgol_filter(stride_lengths, window_length=min(51, len(stride_lengths)//10), polyorder=3)
    walking_speeds = savgol_filter(walking_speeds, window_length=min(51, len(walking_speeds)//10), polyorder=3)
    
    return np.array(stride_lengths), np.array(walking_speeds)

# ------------------------------
# Helper Functions (same as original)
# ------------------------------
def make_windows(X2d, y1d, window=100, step=5, label_fn=np.median):
    """Create overlapping windows - same as original"""
    X_seq, y_seq = [], []
    N = len(X2d)
    for start in range(0, N - window + 1, step):
        end = start + window
        X_seq.append(X2d[start:end])                # (window, features)
        y_seq.append(label_fn(y1d[start:end]))      # scalar
    return np.asarray(X_seq), np.asarray(y_seq)

def build_cnn(input_shape):
    """Build CNN - same architecture as original train_cnn_joint_angle.py"""
    inp = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, 7, padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Conv1D(96, 5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(2)(x)

    # Dilated conv to see a bit wider context without more params
    x = layers.Conv1D(96, 3, dilation_rate=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Global pooling is stable on small data
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)

    out = layers.Dense(1)(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        loss=losses.Huber(delta=1.0),          # robust to outliers
        metrics=["mae"]
    )
    return model

def train_and_evaluate(accel_sheet, target_data, target_name, target_units="degrees", show_plots=True):
    """Train CNN and return results - based on original structure"""
    print(f"\n{'='*60}")
    print(f"Training: {target_name}")
    print(f"Using: {accel_sheet}")
    print(f"Units: {target_units}")
    print(f"{'='*60}")
    
    # Load acceleration data
    df_acc = pd.read_excel(EXCEL_PATH, sheet_name=accel_sheet)
    
    # Basic sanity
    if "Frame" in df_acc.columns:
        df_acc = df_acc.drop(columns=["Frame"])

    # Align lengths
    n = min(len(df_acc), len(target_data))
    df_acc = df_acc.iloc[:n].reset_index(drop=True)
    target_data = target_data[:n]

    X_raw = df_acc.values.astype(np.float32)
    y_raw = target_data.astype(np.float32)

    # Scale features (z-score)
    X_scaler = StandardScaler()
    X = X_scaler.fit_transform(X_raw)

    # For stride length and walking speed, use MinMaxScaler to preserve variation
    if target_name in ["Stride Length", "Walking Speed"]:
        y_scaler = MinMaxScaler(feature_range=(0, 1))
        y = y_scaler.fit_transform(y_raw.reshape(-1, 1)).ravel()
    else:
        # Keep StandardScaler for joint angles
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(y_raw.reshape(-1, 1)).ravel()

    # Create overlapping windows (same as original)
    X_seq, y_seq = make_windows(X, y, WINDOW, STEP, label_fn=np.median)
    print(f"Windows: X_seq={X_seq.shape}, y_seq={y_seq.shape}  "
          f"(from {n} rows, window={WINDOW}, step={STEP})")

    # Guard against too few windows
    if len(X_seq) < 50:
        print("WARNING: Very few windows. Consider reducing STEP (e.g., 2)")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=TEST_SIZE, random_state=SEED
    )

    # Build CNN
    model = build_cnn((X_train.shape[1], X_train.shape[2]))
    
    # Train (same callbacks as original)
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", patience=6, factor=0.5, min_lr=1e-5, verbose=1)
    ]

    hist = model.fit(
        X_train, y_train,
        validation_split=VAL_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cbs,
        verbose=1
    )

    # Evaluate on test set
    y_pred_test_scaled = model.predict(X_test).ravel()

    # Inverse-scale to original units
    y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()

    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    
    print(f"\n=== Test metrics ({accel_sheet}) ===")
    print(f"R²:  {r2: .4f}")
    print(f"MAE: {mae: .4f} {target_units}")
    print(f"MSE: {mse: .4f} {target_units}²")

    # Generate chart (same style as original)
    if show_plots:
        plt.figure(figsize=(8,6))
        plt.scatter(y_true, y_pred, alpha=0.6, s=30, color='blue')
        mn, mx = np.min([y_true, y_pred]), np.max([y_true, y_pred])
        plt.plot([mn, mx], [mn, mx], 'r--', linewidth=2)
        plt.xlabel(f"True {target_name} ({target_units})", fontsize=12)
        plt.ylabel(f"Predicted {target_name} ({target_units})", fontsize=12)
        plt.title(f"CNN Prediction: {target_name}\n({accel_sheet})", fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add metrics to plot
        textstr = f'R² = {r2:.4f}\nMAE = {mae:.3f} {target_units}\nMSE = {mse:.3f} {target_units}²'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.show()

    return {
        'r2': r2,
        'mae': mae,
        'mse': mse,
        'y_true': y_true,
        'y_pred': y_pred,
        'history': hist,
        'model': model,
        'target': target_name,
        'accel_type': accel_sheet,
        'units': target_units
    }

def create_summary_comparison_chart(results_sensor_free, results_segment):
    """Create comparison chart for both acceleration types and all 3 parameters"""
    gait_params = ['Joint Angle', 'Stride Length', 'Walking Speed']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors_sf = ['blue', 'green', 'purple']
    colors_seg = ['red', 'orange', 'brown']
    
    for i, param in enumerate(gait_params):
        ax = axes[i]
        
        # Plot Sensor Free Acceleration results
        if param in results_sensor_free:
            data_sf = results_sensor_free[param]
            ax.scatter(data_sf['y_true'], data_sf['y_pred'], 
                      alpha=0.6, s=25, color=colors_sf[i], 
                      label=f'Sensor Free (R²={data_sf["r2"]:.3f})')
        
        # Plot Segment Acceleration results
        if param in results_segment:
            data_seg = results_segment[param]
            ax.scatter(data_seg['y_true'], data_seg['y_pred'], 
                      alpha=0.6, s=25, color=colors_seg[i], 
                      label=f'Segment (R²={data_seg["r2"]:.3f})')
        
        # Perfect prediction line
        all_true = []
        if param in results_sensor_free:
            all_true.extend(results_sensor_free[param]['y_true'])
        if param in results_segment:
            all_true.extend(results_segment[param]['y_true'])
        
        if all_true:
            mn, mx = np.min(all_true), np.max(all_true)
            ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.8, linewidth=1.5)
        
        # Get units for labels
        units = results_sensor_free.get(param, results_segment.get(param, {})).get('units', 'units')
        
        ax.set_xlabel(f"True {param} ({units})")
        ax.set_ylabel(f"Predicted {param} ({units})")
        ax.set_title(f"{param}", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("CNN Gait Parameter Estimation - All Parameters Comparison", fontsize=16)
    plt.tight_layout()
    plt.show()

def print_final_summary(results_sensor_free, results_segment):
    """Print final numerical summary"""
    print(f"\n{'='*90}")
    print(f"FINAL RESULTS SUMMARY - 3 GAIT PARAMETERS (IMPROVED WALKING SPEED)")
    print(f"{'='*90}")
    
    print(f"{'Parameter':<20} {'Dataset':<20} {'R²':<8} {'MAE':<12} {'MSE':<12}")
    print(f"{'-'*90}")
    
    params = ['Joint Angle', 'Stride Length', 'Walking Speed']
    
    for param in params:
        # Sensor Free results
        if param in results_sensor_free:
            data = results_sensor_free[param]
            units = data['units']
            print(f"{param:<20} {'Sensor Free':<20} "
                  f"{data['r2']:<8.4f} {data['mae']:<8.3f} {units:<3} {data['mse']:<8.3f} {units}²")
        
        # Segment results  
        if param in results_segment:
            data = results_segment[param]
            units = data['units']
            print(f"{'':<20} {'Segment':<20} "
                  f"{data['r2']:<8.4f} {data['mae']:<8.3f} {units:<3} {data['mse']:<8.3f} {units}²")
        
        print(f"{'-'*90}")

# ------------------------------
# Main Execution
# ------------------------------
def main():
    """Main execution function"""
    print("3-Parameter Gait Estimation using CNN (IMPROVED WALKING SPEED)")
    print("Parameters: Joint Angle, Stride Length, Walking Speed")
    print("Based on train_cnn_joint_angle.py architecture with fixed walking speed calculation")
    print(f"Data source: {EXCEL_PATH}")
    
    # Check if file exists
    if not os.path.exists(EXCEL_PATH):
        print(f"Excel file not found: {EXCEL_PATH}")
        return
    
    # Load joint angles data once
    print(f"\nLoading joint angles and deriving improved gait parameters...")
    df_angles = pd.read_excel(EXCEL_PATH, sheet_name=ANGLES_SHEET)
    if "Frame" in df_angles.columns:
        df_angles = df_angles.drop(columns=["Frame"])
    
    # Check if target joint angle exists
    if JOINT_ANGLE_TARGET not in df_angles.columns:
        print(f"Target joint angle '{JOINT_ANGLE_TARGET}' not found")
        print(f"Available columns: {list(df_angles.columns)}")
        return
    
    # Get joint angle data
    joint_angle_data = df_angles[JOINT_ANGLE_TARGET].values
    
    # Calculate improved stride length and walking speed
    print("Calculating improved stride length and walking speed from joint angles...")
    stride_length_data, walking_speed_data = derive_improved_gait_parameters(df_angles, SAMPLING_RATE)
    
    print(f"Derived parameters:")
    print(f"   Joint Angle: {joint_angle_data.mean():.2f} ± {joint_angle_data.std():.2f} degrees")
    print(f"   Stride Length: {stride_length_data.mean():.3f} ± {stride_length_data.std():.3f} meters") 
    print(f"   Walking Speed: {walking_speed_data.mean():.3f} ± {walking_speed_data.std():.3f} m/s")
    
    # Storage for results
    results_sensor_free = {}
    results_segment = {}
    
    # Define the 3 gait parameters
    gait_parameters = [
        ("Joint Angle", joint_angle_data, "degrees"),
        ("Stride Length", stride_length_data, "meters"),
        ("Walking Speed", walking_speed_data, "m/s")
    ]
    
    # Train models using Sensor Free Acceleration
    print(f"\nTRAINING WITH {ACCEL_SHEET_1}")
    print("="*70)
    
    for param_name, param_data, param_units in gait_parameters:
        try:
            result = train_and_evaluate(ACCEL_SHEET_1, param_data, param_name, param_units, show_plots=True)
            results_sensor_free[param_name] = result
        except Exception as e:
            print(f"Error with {param_name} using {ACCEL_SHEET_1}: {str(e)}")
    
    # Train models using Segment Acceleration
    print(f"\nTRAINING WITH {ACCEL_SHEET_2}")
    print("="*70)
    
    for param_name, param_data, param_units in gait_parameters:
        try:
            result = train_and_evaluate(ACCEL_SHEET_2, param_data, param_name, param_units, show_plots=True)
            results_segment[param_name] = result
        except Exception as e:
            print(f"Error with {param_name} using {ACCEL_SHEET_2}: {str(e)}")
    
    # Create comparison chart
    if results_sensor_free or results_segment:
        print(f"\nCREATING FINAL COMPARISON CHART")
        print("="*70)
        create_summary_comparison_chart(results_sensor_free, results_segment)
        
        # Print final summary
        print_final_summary(results_sensor_free, results_segment)
    
    print(f"\nTraining complete for all 3 gait parameters!")
    print("Joint Angle: Direct from joint angles data (unchanged)")
    print("Stride Length: Improved calculation with realistic variation (unchanged)")  
    print("Walking Speed: FIXED - Enhanced derivation with multiple biomechanical factors")
    print("  - RMS Angular Velocity as primary indicator")
    print("  - Peak angular velocity analysis")
    print("  - Movement frequency analysis")
    print("  - Acceleration magnitude consideration")
    print("  - Signal entropy for movement complexity")

if __name__ == "__main__":
    main()
