import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Folder path
folder_path = r"E:\Desktop\Gait_Estimation"
all_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

all_data = []

# Loop through each file
for file in all_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)

    # Drop index column if present
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    # Ensure activity column exists
    if 'activity' in df.columns:
        all_data.append(df)

# Combine all data
full_df = pd.concat(all_data, ignore_index=True)
print("Combined shape:", full_df.shape)

# Label encode activities
le = LabelEncoder()
full_df['activity_id'] = le.fit_transform(full_df['activity'])
activity_map = dict(zip(le.classes_, le.transform(le.classes_)))
print("Activity labels:", activity_map)

# Separate features and labels
X = full_df.drop(columns=['activity', 'activity_id'])  # all 38 sensor columns
y = full_df['activity_id']

# Normalize using MinMax
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Final dataset ready
print(f"Total Samples: {X_scaled.shape[0]} | Features: {X_scaled.shape[1]}")
