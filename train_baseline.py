import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("Starting baseline training pipeline...", flush=True)

# 1. Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['is_train'] = 1
test['is_train'] = 0
df = pd.concat([train, test], ignore_index=True)

# 2. Preprocessing & Normalization
df['product_category'] = df['product_category'].str.strip().str.title()
sku_weight = df.groupby('product_code')['product_weight_kg'].transform('mean')
cat_weight = df.groupby('product_category')['product_weight_kg'].transform('median')
df['product_weight_kg'] = df['product_weight_kg'].fillna(sku_weight).fillna(cat_weight)

df['shelf_visibility'] = df['shelf_visibility'].replace(0.0, np.nan)
sku_vis = df.groupby('product_code')['shelf_visibility'].transform('mean')
cat_vis = df.groupby('product_category')['shelf_visibility'].transform('median')
df['shelf_visibility'] = df['shelf_visibility'].fillna(sku_vis).fillna(cat_vis)

df['store_size'] = df['store_size'].fillna('Unknown')
non_food_cats = ['Health And Hygiene', 'Household', 'Others']
df.loc[df['product_category'].isin(non_food_cats), 'fat_content'] = 'Non-Edible'

# 3. Feature Engineering
df['price_per_kg'] = df['product_price'] / df['product_weight_kg']
cat_price_mean = df.groupby('product_category')['product_price'].transform('mean')
cat_price_std = df.groupby('product_category')['product_price'].transform('std')
df['price_to_cat_mean'] = df['product_price'] / (cat_price_mean + 1e-5)
df['price_cat_zscore'] = (df['product_price'] - cat_price_mean) / (cat_price_std + 1e-5)

store_price_mean = df.groupby('store_code')['product_price'].transform('mean')
df['price_to_store_mean'] = df['product_price'] / (store_price_mean + 1e-5)

sku_vis_mean = df.groupby('product_code')['shelf_visibility'].transform('mean')
df['vis_to_sku_mean'] = df['shelf_visibility'] / (sku_vis_mean + 1e-5)
cat_vis_mean = df.groupby('product_category')['shelf_visibility'].transform('mean')
df['vis_to_cat_mean'] = df['shelf_visibility'] / (cat_vis_mean + 1e-5)
store_vis_mean = df.groupby('store_code')['shelf_visibility'].transform('mean')
df['vis_to_store_mean'] = df['shelf_visibility'] / (store_vis_mean + 1e-5)

df['product_store_count'] = df.groupby('product_code')['store_code'].transform('nunique')
df['store_product_count'] = df.groupby('store_code')['product_code'].transform('nunique')
df['is_corner_shop'] = (df['store_format'] == 'Corner Shop').astype(int)

df['store_format_tier'] = df['store_format'] + '_' + df['store_location_tier']
df['store_format_size'] = df['store_format'] + '_' + df['store_size']
df['store_cat_interaction'] = df['store_code'] + '_' + df['product_category']
df['product_code_prefix'] = df['product_code'].str.split('-').str[1].str[0]

def assign_price_tier(p):
    if p < 69: return 0
    elif p < 136: return 1
    elif p < 203: return 2
    else: return 3
df['price_tier'] = df['product_price'].apply(assign_price_tier)

# Split back to train and test
train_df = df[df['is_train'] == 1].reset_index(drop=True)
test_df = df[df['is_train'] == 0].reset_index(drop=True)

cat_cols = ['product_code', 'fat_content', 'product_category', 'store_code', 
            'store_size', 'store_location_tier', 'store_format', 'store_format_tier', 
            'store_format_size', 'store_cat_interaction', 'product_code_prefix']

for col in cat_cols:
    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')

feature_cols = [c for c in train_df.columns if c not in ['id', 'total_sales', 'is_train']]

print(f"Dataset ready. Total features: {len(feature_cols)}", flush=True)

# 4. 5-Fold Cross Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))

X_train = train_df[feature_cols].copy()
y_train = train_df['total_sales'].values

print("--- Starting 5-Fold Cross Validation ---", flush=True)
for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_trn, y_trn = X_train.iloc[trn_idx].copy(), y_train[trn_idx]
    X_val, y_val = X_train.iloc[val_idx].copy(), y_train[val_idx]
    
    # Target encoding strictly on train fold
    tmp = pd.DataFrame({'store': X_trn['store_code'], 'cat': X_trn['product_category'], 'store_cat': X_trn['store_cat_interaction'], 'target': y_trn})
    sm = tmp.groupby('store', observed=False)['target'].mean()
    cm = tmp.groupby('cat', observed=False)['target'].mean()
    scm = tmp.groupby('store_cat', observed=False)['target'].mean()
    ov = y_trn.mean()
    
    for d in [X_trn, X_val]:
        d['te_store'] = d['store_code'].map(sm).fillna(ov).astype(float)
        d['te_cat'] = d['product_category'].map(cm).fillna(ov).astype(float)
        d['te_store_cat'] = d['store_cat_interaction'].map(scm).fillna(ov).astype(float)
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1,
        n_jobs=4
    )
    lgb_model.fit(
        X_trn, y_trn,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    val_pred_lgb = lgb_model.predict(X_val)
    oof_lgb[val_idx] = val_pred_lgb
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        random_state=42 + fold,
        tree_method='hist',
        n_jobs=4
    )
    xgb_model.fit(
        X_trn, y_trn,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    val_pred_xgb = xgb_model.predict(X_val)
    oof_xgb[val_idx] = val_pred_xgb
    
    rmse_lgb = root_mean_squared_error(y_val, val_pred_lgb)
    rmse_xgb = root_mean_squared_error(y_val, val_pred_xgb)
    print(f"Fold {fold+1} | LightGBM RMSE: {rmse_lgb:.4f} | XGBoost RMSE: {rmse_xgb:.4f}", flush=True)

overall_lgb_rmse = root_mean_squared_error(y_train, oof_lgb)
overall_xgb_rmse = root_mean_squared_error(y_train, oof_xgb)

# Optimal blend weight search
weights = np.linspace(0, 1, 101)
best_w = 0.5
best_blend_rmse = 999999
for w in weights:
    blend = w * oof_lgb + (1 - w) * oof_xgb
    score = root_mean_squared_error(y_train, blend)
    if score < best_blend_rmse:
        best_blend_rmse = score
        best_w = w

print("\n================ CROSS-VALIDATION SUMMARY ================", flush=True)
print(f"Overall Out-Of-Fold LightGBM RMSE : {overall_lgb_rmse:.4f}", flush=True)
print(f"Overall Out-Of-Fold XGBoost RMSE  : {overall_xgb_rmse:.4f}", flush=True)
print(f"Optimized Blend RMSE (Weight: {best_w:.2f} LGB / {1-best_w:.2f} XGB): {best_blend_rmse:.4f}", flush=True)

