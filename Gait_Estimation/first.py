import pandas as pd

# === Path to your Excel file ===
excel_path = r"E:\Desktop\Gait_Estimation\New Session-137.xlsx"

# === Sheets to inspect ===
sheets_to_check = [
    "Joint Angles ZXY",
    "Sensor Free Acceleration",
    "Segment Acceleration"
]

# === Loop through each and display info ===
for sheet in sheets_to_check:
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet)
        print(f"\n✅ Loaded sheet: {sheet}")
        print(f"Shape: {df.shape}")
        print("First 5 columns:")
        for i, col in enumerate(df.columns[:5]):
            print(f"  {i+1}. {col}")
    except Exception as e:
        print(f"\n❌ Error loading sheet: {sheet}")
        print(f"   {e}")
