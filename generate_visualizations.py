import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

os.makedirs('assets', exist_ok=True)
print("--- Generating Publication Quality Visualizations ---", flush=True)

# 1. Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['is_train'] = 1
test['is_train'] = 0
df = pd.concat([train, test], ignore_index=True)

# Preprocessing & Imputation
df['product_category'] = df['product_category'].astype(str).str.strip().str.title()
sku_weight = df.groupby('product_code')['product_weight_kg'].transform('mean')
cat_weight = df.groupby('product_category')['product_weight_kg'].transform('median')
df['product_weight_kg'] = df['product_weight_kg'].fillna(sku_weight).fillna(cat_weight)

df['shelf_visibility'] = df['shelf_visibility'].replace(0.0, np.nan)
sku_vis = df.groupby('product_code')['shelf_visibility'].transform('mean')
cat_vis = df.groupby('product_category')['shelf_visibility'].transform('median')
df['shelf_visibility'] = df['shelf_visibility'].fillna(sku_vis).fillna(cat_vis)

df['store_size'] = df['store_size'].fillna('Unknown')
non_food = ['Health And Hygiene', 'Household', 'Others']
df.loc[df['product_category'].isin(non_food), 'fat_content'] = 'Non-Edible'

# Feature Engineering
df['price_per_kg'] = df['product_price'] / df['product_weight_kg']
cat_p_mean = df.groupby('product_category')['product_price'].transform('mean')
cat_p_std = df.groupby('product_category')['product_price'].transform('std')
df['price_to_cat_mean'] = df['product_price'] / (cat_p_mean + 1e-5)
df['price_cat_zscore'] = (df['product_price'] - cat_p_mean) / (cat_p_std + 1e-5)

store_p_mean = df.groupby('store_code')['product_price'].transform('mean')
df['price_to_store_mean'] = df['product_price'] / (store_p_mean + 1e-5)

sku_v_mean = df.groupby('product_code')['shelf_visibility'].transform('mean')
df['vis_to_sku_mean'] = df['shelf_visibility'] / (sku_v_mean + 1e-5)
cat_v_mean = df.groupby('product_category')['shelf_visibility'].transform('mean')
df['vis_to_cat_mean'] = df['shelf_visibility'] / (cat_v_mean + 1e-5)
store_v_mean = df.groupby('store_code')['shelf_visibility'].transform('mean')
df['vis_to_store_mean'] = df['shelf_visibility'] / (store_v_mean + 1e-5)

df['product_store_count'] = df.groupby('product_code')['store_code'].transform('nunique')
df['store_product_count'] = df.groupby('store_code')['product_code'].transform('nunique')
df['is_corner_shop'] = (df['store_format'] == 'Corner Shop').astype(int)

df['store_format_tier'] = df['store_format'].astype(str) + '_' + df['store_location_tier'].astype(str)
df['store_format_size'] = df['store_format'].astype(str) + '_' + df['store_size'].astype(str)
df['store_cat_interaction'] = df['store_code'].astype(str) + '_' + df['product_category'].astype(str)
df['product_code_prefix'] = df['product_code'].astype(str).str.split('-').str[1].str[0]

def assign_price_tier(p):
    if p < 69: return 0
    elif p < 136: return 1
    elif p < 203: return 2
    else: return 3
df['price_tier'] = df['product_price'].apply(assign_price_tier)

train_df = df[df['is_train'] == 1].reset_index(drop=True)
test_df = df[df['is_train'] == 0].reset_index(drop=True)

cat_cols = ['product_code', 'fat_content', 'product_category', 'store_code', 
            'store_size', 'store_location_tier', 'store_format', 'store_format_tier', 
            'store_format_size', 'store_cat_interaction', 'product_code_prefix']
feature_cols = [c for c in train_df.columns if c not in ['id', 'total_sales', 'is_train']]

# -------------------------------------------------------------
# PLOT 1: Store Format vs Total Sales & Estimated Units Sold
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

order = train_df.groupby('store_format')['total_sales'].median().sort_values(ascending=False).index
sns.boxplot(data=train_df, x='store_format', y='total_sales', order=order, palette='mako', ax=axes[0])
axes[0].set_title('Total Sales Distribution by Store Format', fontsize=14, weight='bold', pad=12)
axes[0].set_xlabel('Store Format', weight='bold')
axes[0].set_ylabel('Total Sales (₦ / $)', weight='bold')
axes[0].tick_params(axis='x', rotation=15)

train_df['units_sold'] = train_df['total_sales'] / train_df['product_price']
sns.barplot(data=train_df, x='store_format', y='units_sold', hue='store_location_tier', palette='viridis', ci=None, ax=axes[1])
axes[1].set_title('Average Units Sold by Store Format & Location Tier', fontsize=14, weight='bold', pad=12)
axes[1].set_xlabel('Store Format', weight='bold')
axes[1].set_ylabel('Average Units Sold per SKU', weight='bold')
axes[1].tick_params(axis='x', rotation=15)
axes[1].legend(title='Location Tier', frameon=True)

plt.tight_layout()
plt.savefig('assets/store_format_sales_distribution.png', dpi=300)
plt.close()
print("Saved assets/store_format_sales_distribution.png", flush=True)

# -------------------------------------------------------------
# PLOT 2: Category Revenue Breakdown & Price Elasticity
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 7))
cat_agg = train_df.groupby('product_category').agg({
    'total_sales': 'sum',
    'product_price': 'mean'
}).reset_index().sort_values('total_sales', ascending=True)

colors = sns.color_palette("crest", len(cat_agg))
bars = ax.barh(cat_agg['product_category'], cat_agg['total_sales'] / 1e6, color=colors)
ax.set_title('Total Revenue Generated by Product Category (in Millions)', fontsize=15, weight='bold', pad=14)
ax.set_xlabel('Total Revenue (Millions ₦)', weight='bold')
ax.set_ylabel('Product Category', weight='bold')

for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.05, bar.get_y() + bar.get_height()/2, f'{width:.2f}M', ha='left', va='center', fontsize=10, weight='bold')

plt.tight_layout()
plt.savefig('assets/category_revenue_breakdown.png', dpi=300)
plt.close()
print("Saved assets/category_revenue_breakdown.png", flush=True)

# -------------------------------------------------------------
# PLOT 3: Train 5-Fold Ensemble & Compute Feature Importances + Actual vs Predicted
# -------------------------------------------------------------
with open('best_models_config.json', 'r') as f:
    config = json.load(f)

best_lgb_params = config['best_lgb_params']
best_xgb_params = config['best_xgb_params']
w_lgb = config['ensemble_weights']['lgb']
w_xgb = config['ensemble_weights']['xgb']

kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_train_raw = train_df[feature_cols].copy()
y_train = train_df['total_sales'].values

oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
feature_importances_lgb = np.zeros(len(feature_cols) + 3) # including 3 target encoding features

all_feat_names = feature_cols + ['te_store', 'te_cat', 'te_store_cat']

for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train_raw, y_train)):
    X_trn = X_train_raw.iloc[trn_idx].copy()
    y_trn = y_train[trn_idx]
    X_val = X_train_raw.iloc[val_idx].copy()
    y_val = y_train[val_idx]
    
    tmp = pd.DataFrame({'store': X_trn['store_code'], 'cat': X_trn['product_category'], 'store_cat': X_trn['store_cat_interaction'], 'target': y_trn})
    sm = tmp.groupby('store', observed=False)['target'].mean()
    cm = tmp.groupby('cat', observed=False)['target'].mean()
    scm = tmp.groupby('store_cat', observed=False)['target'].mean()
    ov = y_trn.mean()
    
    for d in [X_trn, X_val]:
        d['te_store'] = d['store_code'].map(sm).fillna(ov).astype(float)
        d['te_cat'] = d['product_category'].map(cm).fillna(ov).astype(float)
        d['te_store_cat'] = d['store_cat_interaction'].map(scm).fillna(ov).astype(float)
        
    for col in cat_cols:
        X_trn[col] = X_trn[col].astype('category')
        X_val[col] = X_val[col].astype('category')
        
    lgb_m = lgb.LGBMRegressor(
        objective='regression', metric='rmse', n_estimators=1500,
        **best_lgb_params, random_state=42, verbose=-1, n_jobs=4
    )
    lgb_m.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(60, verbose=False)])
    oof_lgb[val_idx] = lgb_m.predict(X_val)
    feature_importances_lgb += lgb_m.feature_importances_ / 5

    # XGBoost
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_trn_xgb = X_trn.copy()
    X_val_xgb = X_val.copy()
    X_trn_xgb[cat_cols] = oe.fit_transform(X_trn_xgb[cat_cols].astype(str))
    X_val_xgb[cat_cols] = oe.transform(X_val_xgb[cat_cols].astype(str))
    
    xgb_m = xgb.XGBRegressor(
        objective='reg:squarederror', eval_metric='rmse', n_estimators=1500,
        early_stopping_rounds=50, **best_xgb_params, random_state=42, tree_method='hist', n_jobs=4
    )
    xgb_m.fit(X_trn_xgb, y_trn, eval_set=[(X_val_xgb, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_m.predict(X_val_xgb)

oof_final = w_lgb * oof_lgb + w_xgb * oof_xgb

# Feature Importance Plot
fi_df = pd.DataFrame({
    'Feature': all_feat_names,
    'Importance': feature_importances_lgb
}).sort_values('Importance', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(data=fi_df, x='Importance', y='Feature', palette='magma', ax=ax)
ax.set_title('Top 15 Most Influential Features (LightGBM 5-Fold Ensemble)', fontsize=15, weight='bold', pad=14)
ax.set_xlabel('Split Importance Score', weight='bold')
ax.set_ylabel('Engineered Feature', weight='bold')
plt.tight_layout()
plt.savefig('assets/feature_importance.png', dpi=300)
plt.close()
print("Saved assets/feature_importance.png", flush=True)

# Actual vs Predicted & Residuals Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Actual vs Predicted
axes[0].scatter(y_train, oof_final, alpha=0.3, color='#2b5c8f', edgecolors='none', s=25)
ideal_line = np.linspace(0, max(y_train), 100)
axes[0].plot(ideal_line, ideal_line, color='#d9534f', linestyle='--', lw=2.5, label='Perfect Prediction')
axes[0].set_title('Out-of-Fold Actual vs Predicted Sales', fontsize=14, weight='bold', pad=12)
axes[0].set_xlabel('Actual Sales (₦)', weight='bold')
axes[0].set_ylabel('Predicted Sales (₦)', weight='bold')
axes[0].legend()

# Subplot 2: Residuals Distribution
residuals = y_train - oof_final
sns.histplot(residuals, kde=True, color='#348888', bins=50, ax=axes[1])
axes[1].axvline(0, color='red', linestyle='--', lw=2)
axes[1].set_title('Residual Error Distribution (Mean ≈ 0)', fontsize=14, weight='bold', pad=12)
axes[1].set_xlabel('Residual Error (Actual - Predicted)', weight='bold')
axes[1].set_ylabel('Count', weight='bold')

plt.tight_layout()
plt.savefig('assets/actual_vs_predicted.png', dpi=300)
plt.close()
print("Saved assets/actual_vs_predicted.png", flush=True)
print("All visualization assets generated successfully!", flush=True)

