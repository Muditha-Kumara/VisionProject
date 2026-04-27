import pandas as pd

# Read the CSV file
df = pd.read_csv('deploy_lane/pc_luckfox_live_metrics.csv')

# Exclude __SUMMARY__ row
df = df[df['image'] != '__SUMMARY__']

# Sort by image ascending
df = df.sort_values(by='image')

# Convert columns to numeric, coercion will turn non-numeric to NaN
numeric_cols = ['pc_detections', 'luckfox_detections', 'Lane_1_pc', 'Lane_1_luckfox', 'Lane_2_pc', 'Lane_2_luckfox', 'Lane_3_pc', 'Lane_3_luckfox']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

new_df = pd.DataFrame()
new_df['image'] = df['image']
new_df['pc_detections'] = df['pc_detections']
new_df['luckfox_detections'] = df['luckfox_detections']
new_df['diff_luckfox_minus_pc'] = df['luckfox_detections'] - df['pc_detections']
new_df['abs_diff'] = new_df['diff_luckfox_minus_pc'].abs()

if 'luckfox_status' in df.columns:
    new_df['luckfox_status'] = df['luckfox_status']
else:
    new_df['luckfox_status'] = 'N/A'

for i in range(1, 4):
    pc_col = f'Lane_{i}_pc'
    lf_col = f'Lane_{i}_luckfox'
    new_df[pc_col] = df[pc_col] if pc_col in df.columns else 0
    new_df[lf_col] = df[lf_col] if lf_col in df.columns else 0

columns_order = [
    'image', 'pc_detections', 'luckfox_detections', 'diff_luckfox_minus_pc', 
    'abs_diff', 'luckfox_status', 'Lane_1_pc', 'Lane_1_luckfox', 
    'Lane_2_pc', 'Lane_2_luckfox', 'Lane_3_pc', 'Lane_3_luckfox'
]
new_df = new_df[columns_order]

new_df.to_csv('deploy_lane/pc_luckfox_detection_comparison.csv', index=False)

print(f"Total rows: {len(new_df)}")
print("First 5 lines:")
print(new_df.head(5).to_string(index=False))
