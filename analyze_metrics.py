import pandas as pd
import numpy as np

file_path = 'deploy_lane/pc_luckfox_live_metrics.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File {file_path} not found.")
    exit(1)

# Exclude __SUMMARY__ row
df = df[df['image'] != '__SUMMARY__']

# Convert detections to numeric, drop non-numeric rows
df['pc_detections'] = pd.to_numeric(df['pc_detections'], errors='coerce')
df['luckfox_detections'] = pd.to_numeric(df['luckfox_detections'], errors='coerce')
df = df.dropna(subset=['pc_detections', 'luckfox_detections'])

num_image_rows = len(df)

# Stats for pc_detections and luckfox_detections
stats = {}
for col in ['pc_detections', 'luckfox_detections']:
    stats[col] = {
        'mean': df[col].mean(),
        'median': df[col].median(),
        'min': df[col].min(),
        'max': df[col].max()
    }

# Error metrics
df['diff'] = df['luckfox_detections'] - df['pc_detections']
df['abs_diff'] = df['diff'].abs()

mae = df['abs_diff'].mean()
msd = df['diff'].mean()

# Comparisons
count_luckfox_gt_pc = (df['luckfox_detections'] > df['pc_detections']).sum()
count_luckfox_lt_pc = (df['luckfox_detections'] < df['pc_detections']).sum()
count_equal = (df['luckfox_detections'] == df['pc_detections']).sum()

pct_luckfox_gt_pc = (count_luckfox_gt_pc / num_image_rows) * 100
pct_luckfox_lt_pc = (count_luckfox_lt_pc / num_image_rows) * 100
pct_equal = (count_equal / num_image_rows) * 100

# Lane-wise sums and means
pc_cols = [c for c in df.columns if c.endswith('_pc')]
luckfox_cols = [c for c in df.columns if c.endswith('_luckfox')]

lane_stats = {}
for col in pc_cols + luckfox_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    lane_stats[col] = {
        'sum': df[col].sum(),
        'mean': df[col].mean()
    }

# Output
print(f"Number of image rows: {num_image_rows}")
print("\nDetection Statistics:")
for col, s in stats.items():
    print(f"{col}: Mean={s['mean']:.2f}, Median={s['median']:.2f}, Min={s['min']:.2f}, Max={s['max']:.2f}")

print(f"\nMean Absolute Error: {mae:.2f}")
print(f"Mean Signed Difference (Luckfox - PC): {msd:.2f}")

print(f"\nComparisons:")
print(f"Luckfox > PC: {count_luckfox_gt_pc} ({pct_luckfox_gt_pc:.2f}%)")
print(f"Luckfox < PC: {count_luckfox_lt_pc} ({pct_luckfox_lt_pc:.2f}%)")
print(f"Equal: {count_equal} ({pct_equal:.2f}%)")

print("\nLane-wise Stats (Top 10 columns by mean):")
sorted_lane_stats = sorted(lane_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
for col, s in sorted_lane_stats[:10]:
    print(f"{col}: Sum={s['sum']:.2f}, Mean={s['mean']:.4f}")

print("\nTop 15 images with largest absolute detection difference:")
top_15 = df.sort_values(by='abs_diff', ascending=False).head(15)
print(top_15[['image', 'pc_detections', 'luckfox_detections', 'diff']].to_string(index=False))
