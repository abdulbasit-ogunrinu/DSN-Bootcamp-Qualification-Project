import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
import optuna
import warnings
import json
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("--- Step 4: Hyperparameter Tuning & Ensembling Pipeline ---", flush=True)

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

kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_train = train_df[feature_cols].copy()
y_train = train_df['total_sales'].values

# Precompute fold splits and target encodings to accelerate Optuna trials
prepared_folds = []
for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_trn, y_trn = X_train.iloc[trn_idx].copy(), y_train[trn_idx]
    X_val, y_val = X_train.iloc[val_idx].copy(), y_train[val_idx]
    
    tmp = pd.DataFrame({'store': X_trn['store_code'], 'cat': X_trn['product_category'], 'store_cat': X_trn['store_cat_interaction'], 'target': y_trn})
    sm = tmp.groupby('store', observed=False)['target'].mean()
    cm = tmp.groupby('cat', observed=False)['target'].mean()
    scm = tmp.groupby('store_cat', observed=False)['target'].mean()
    ov = y_trn.mean()
    
    for d in [X_trn, X_val]:
        d['te_store'] = d['store_code'].map(sm).fillna(ov).astype(float)
        d['te_cat'] = d['product_category'].map(cm).fillna(ov).astype(float)
        d['te_store_cat'] = d['store_cat_interaction'].map(scm).fillna(ov).astype(float)
    
    prepared_folds.append((X_trn, y_trn, X_val, y_val, val_idx))

print("Data prepared and CV folds configured.", flush=True)

# 4. Optuna Tuning for LightGBM
best_lgb_params = {
    'learning_rate': 0.02878,
    'num_leaves': 28,
    'max_depth': 3,
    'min_child_samples': 52,
    'subsample': 0.8978,
    'colsample_bytree': 0.5447,
    'reg_alpha': 0.7570,
    'reg_lambda': 0.00267
}
print(f"\nTuned LightGBM Params: {best_lgb_params}", flush=True)

# 5. Optuna Tuning for XGBoost (with early stopping)
print("\n>>> Tuning XGBoost with Optuna (15 Fast Trials)...", flush=True)

def objective_xgb(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': 600,
        'early_stopping_rounds': 40,
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.08, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.85),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
        'enable_categorical': True,
        'random_state': 42,
        'tree_method': 'hist',
        'n_jobs': 4
    }
    
    oof = np.zeros(len(train_df))
    for X_trn, y_trn, X_val, y_val, val_idx in prepared_folds:
        model = xgb.XGBRegressor(**params)
        model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=False)
        oof[val_idx] = model.predict(X_val)
    
    return root_mean_squared_error(y_train, oof)

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=15)
print(f"Best XGBoost Trial RMSE: {study_xgb.best_value:.4f}", flush=True)
print(f"Best XGBoost Params: {study_xgb.best_params}", flush=True)

# 6. Train Final 5-Fold Tuned Models
print("\n>>> Training Final 5-Fold Tuned Models...", flush=True)
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
    **study_xgb.best_params,
    'enable_categorical': True,
    'random_state': 42,
    'tree_method': 'hist',
    'n_jobs': 4
}

oof_tuned_lgb = np.zeros(len(train_df))
oof_tuned_xgb = np.zeros(len(train_df))

for fold, (X_trn, y_trn, X_val, y_val, val_idx) in enumerate(prepared_folds):
    # LightGBM
    lgb_m = lgb.LGBMRegressor(**full_lgb_params)
    lgb_m.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(60, verbose=False)])
    oof_tuned_lgb[val_idx] = lgb_m.predict(X_val)
    
    # XGBoost
    xgb_m = xgb.XGBRegressor(**full_xgb_params)
    xgb_m.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=False)
    oof_tuned_xgb[val_idx] = xgb_m.predict(X_val)
    
    fold_lgb_err = root_mean_squared_error(y_val, oof_tuned_lgb[val_idx])
    fold_xgb_err = root_mean_squared_error(y_val, oof_tuned_xgb[val_idx])
    print(f"Fold {fold+1} | Tuned LightGBM RMSE: {fold_lgb_err:.4f} | Tuned XGBoost RMSE: {fold_xgb_err:.4f}", flush=True)

tuned_lgb_rmse = root_mean_squared_error(y_train, oof_tuned_lgb)
tuned_xgb_rmse = root_mean_squared_error(y_train, oof_tuned_xgb)

# 7. Optimize Ensemble Weights
def ensemble_loss(w):
    w1 = w[0]
    w2 = 1.0 - w1
    pred = w1 * oof_tuned_lgb + w2 * oof_tuned_xgb
    return root_mean_squared_error(y_train, pred)

res = minimize(ensemble_loss, [0.8], bounds=[(0, 1)], method='Nelder-Mead')
w_lgb = float(res.x[0])
w_xgb = 1.0 - w_lgb
final_ensemble_oof = w_lgb * oof_tuned_lgb + w_xgb * oof_tuned_xgb
final_ensemble_rmse = root_mean_squared_error(y_train, final_ensemble_oof)

print("\n================ STEP 4 ENSEMBLING SUMMARY ================", flush=True)
print(f"Tuned LightGBM OOF RMSE : {tuned_lgb_rmse:.4f}", flush=True)
print(f"Tuned XGBoost OOF RMSE  : {tuned_xgb_rmse:.4f}", flush=True)
print(f"Optimized Blend Weights : {w_lgb:.3f} LightGBM + {w_xgb:.3f} XGBoost", flush=True)
print(f"Final Ensemble OOF RMSE : {final_ensemble_rmse:.4f}", flush=True)

# Save best configurations
config = {
    'best_lgb_params': best_lgb_params,
    'best_xgb_params': study_xgb.best_params,
    'ensemble_weights': {'lgb': w_lgb, 'xgb': w_xgb},
    'final_rmse': float(final_ensemble_rmse)
}
with open('best_models_config.json', 'w') as f:
    json.dump(config, f, indent=4)
print("Saved best_models_config.json successfully.", flush=True)

