import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("--- Step 5: Generating Final Kaggle Submission ---", flush=True)

# 1. Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

train['is_train'] = 1
test['is_train'] = 0
df = pd.concat([train, test], ignore_index=True)

# 2. Preprocessing & Imputation
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

feature_cols = [c for c in train_df.columns if c not in ['id', 'total_sales', 'is_train']]

# Load best config
with open('best_models_config.json', 'r') as f:
    config = json.load(f)

best_lgb_params = config['best_lgb_params']
best_xgb_params = config['best_xgb_params']
w_lgb = config['ensemble_weights']['lgb']
w_xgb = config['ensemble_weights']['xgb']

print(f"Ensemble configuration: {w_lgb:.3f} LightGBM + {w_xgb:.3f} XGBoost", flush=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_train_raw = train_df[feature_cols].copy()
y_train = train_df['total_sales'].values
X_test_raw = test_df[feature_cols].copy()

test_preds_lgb = np.zeros(len(test_df))
test_preds_xgb = np.zeros(len(test_df))
oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))

full_lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 1500,
    **best_lgb_params,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': 4
}

full_xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': 1500,
    'early_stopping_rounds': 50,
    **best_xgb_params,
    'random_state': 42,
    'tree_method': 'hist',
    'n_jobs': 4
}

print("\n>>> Training 5 Folds & Predicting on Test Set...", flush=True)

for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train_raw, y_train)):
    # 1. Prepare fold data
    X_trn = X_train_raw.iloc[trn_idx].copy()
    y_trn = y_train[trn_idx]
    X_val = X_train_raw.iloc[val_idx].copy()
    y_val = y_train[val_idx]
    X_tst = X_test_raw.copy()
    
    # 2. Target encoding strictly from fold training split
    tmp = pd.DataFrame({'store': X_trn['store_code'], 'cat': X_trn['product_category'], 'store_cat': X_trn['store_cat_interaction'], 'target': y_trn})
    sm = tmp.groupby('store', observed=False)['target'].mean()
    cm = tmp.groupby('cat', observed=False)['target'].mean()
    scm = tmp.groupby('store_cat', observed=False)['target'].mean()
    ov = y_trn.mean()
    
    for d in [X_trn, X_val, X_tst]:
        d['te_store'] = d['store_code'].map(sm).fillna(ov).astype(float)
        d['te_cat'] = d['product_category'].map(cm).fillna(ov).astype(float)
        d['te_store_cat'] = d['store_cat_interaction'].map(scm).fillna(ov).astype(float)
    
    # 3. LightGBM (with category dtype)
    X_trn_lgb, X_val_lgb, X_tst_lgb = X_trn.copy(), X_val.copy(), X_tst.copy()
    for col in cat_cols:
        X_trn_lgb[col] = X_trn_lgb[col].astype('category')
        X_val_lgb[col] = X_val_lgb[col].astype('category')
        X_tst_lgb[col] = X_tst_lgb[col].astype('category')
        
    lgb_m = lgb.LGBMRegressor(**full_lgb_params)
    lgb_m.fit(X_trn_lgb, y_trn, eval_set=[(X_val_lgb, y_val)], callbacks=[lgb.early_stopping(60, verbose=False)])
    oof_lgb[val_idx] = lgb_m.predict(X_val_lgb)
    test_preds_lgb += lgb_m.predict(X_tst_lgb) / kf.n_splits
    
    # 4. XGBoost (with OrdinalEncoder handling unseen test categories)
    X_trn_xgb, X_val_xgb, X_tst_xgb = X_trn.copy(), X_val.copy(), X_tst.copy()
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_trn_xgb[cat_cols] = oe.fit_transform(X_trn_xgb[cat_cols].astype(str))
    X_val_xgb[cat_cols] = oe.transform(X_val_xgb[cat_cols].astype(str))
    X_tst_xgb[cat_cols] = oe.transform(X_tst_xgb[cat_cols].astype(str))
    
    xgb_m = xgb.XGBRegressor(**full_xgb_params)
    xgb_m.fit(X_trn_xgb, y_trn, eval_set=[(X_val_xgb, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_m.predict(X_val_xgb)
    test_preds_xgb += xgb_m.predict(X_tst_xgb) / kf.n_splits
    
    print(f"Fold {fold+1} complete | LGB RMSE: {root_mean_squared_error(y_val, oof_lgb[val_idx]):.4f} | XGB RMSE: {root_mean_squared_error(y_val, oof_xgb[val_idx]):.4f}", flush=True)

# 5. Final Ensemble Predictions
final_test_preds = w_lgb * test_preds_lgb + w_xgb * test_preds_xgb

# Minimum boundary clipping
min_train_sales = float(y_train.min())
final_test_preds = np.clip(final_test_preds, a_min=min_train_sales, a_max=None)

# 6. Format & Validate Submission File
submission = pd.DataFrame({
    'id': test_df['id'],
    'total_sales': np.round(final_test_preds, 4)
})

# Validation Assertions
assert submission.shape == sample_sub.shape, f"Shape mismatch: {submission.shape} vs {sample_sub.shape}"
assert (submission['id'] == sample_sub['id']).all(), "ID column ordering mismatch!"
assert submission['total_sales'].isnull().sum() == 0, "Submission contains NaN values!"
assert (submission['total_sales'] < 0).sum() == 0, "Submission contains negative sales!"

submission.to_csv('submission.csv', index=False)
print("\n>>> Submission successfully generated and verified!", flush=True)
print(f"Saved: submission.csv | Rows: {len(submission)}")
print("\nSample Predictions (Head 10):")
print(submission.head(10))

print("\nSummary Statistics of Predictions vs Train Target:")
stats = pd.DataFrame({
    'Train total_sales': train['total_sales'].describe(),
    'Predicted total_sales': submission['total_sales'].describe()
})
print(stats)

