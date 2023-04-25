from molecular_features import rdkit_2d_descriptors_generator
from rdkit import Chem
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib

def glb_descriptors(dataset):
    g_desc = []
    for smiles in tqdm(dataset):
        mol = Chem.MolFromSmiles(smiles)
        g_desc.append(list(rdkit_2d_descriptors_generator(mol)))
    return g_desc

dataset = pd.read_csv('./data_files/Final_Dataset.csv')
Huang_dataset = pd.read_csv('./data_files/Huang & Massa density.csv')

dataset = np.array(dataset)
Huang_dataset = np.array(Huang_dataset)

X = glb_descriptors(dataset[:,2])
# X = np.array(X)
# np.save('./RF/feature.npy', X)
# X = np.load('./RF/feature.npy')
X = np.nan_to_num(np.float32(X))
y = dataset[:,1]

X_Huang_test = glb_descriptors(Huang_dataset[:,2])
# X_Huang_test = np.array(X_Huang_test)
# np.save('./RF/feature.npy', X_Huang_test)
# X_Huang_test = np.load('./RF/feature_Huang.npy')
X_Huang_test = np.nan_to_num(np.float32(X_Huang_test))
y_Huang_test = Huang_dataset[:,1]

seed = 42

r2_score = []
mae = []
rmse = []
r2_score_Huang = []
mae_Huang = []
rmse_Huang = []

# Traning the RF model
for i in tqdm(range(3)):
    X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.1)

    rf = RandomForestRegressor( n_estimators=1200, min_samples_split=5, min_samples_leaf=2,
                                max_features='sqrt', max_depth=20, bootstrap=False,
                                n_jobs=-1, random_state=42+i, criterion='absolute_error')
    rf.fit(X_train, y_train)

    # joblib.dump(rf, f"rf_model_{i}.joblib") # save the model

    pred = rf.predict(X_test)
    r2_score.append(rf.score(X_test, y_test))
    mae.append(mean_absolute_error(pred, y_test))
    rmse.append(mean_squared_error(pred, y_test,squared=False))
    
    Huang_pred = rf.predict(X_Huang_test)
    r2_score_Huang.append(rf.score(X_Huang_test, y_Huang_test))
    mae_Huang.append(mean_absolute_error(Huang_pred, y_Huang_test))
    rmse_Huang.append(mean_squared_error(Huang_pred, y_Huang_test,squared=False))

    # save the result
    # np.save(f"X_test_Huang_{i}.npy", X_test)
    # np.save(f"pred_Huang_{i}.npy", pred)
    # np.save(f"y_test_Huang_{i}.npy", y_test)

print(f'mae = {np.mean(mae):.4f} ± {np.std(mae):.4f}')
print(f'rmse = {np.mean(rmse):.4f} ± {np.std(rmse):.4f}')
print(f'r2_score = {np.mean(r2_score):.4f} ± {np.std(r2_score):.4f}')

print(f'mae_Huang = {np.mean(mae_Huang):.4f} ± {np.std(mae_Huang):.4f}')
print(f'rmse_Huang = {np.mean(rmse_Huang):.4f} ± {np.std(rmse_Huang):.4f}')
print(f'r2_score_Huang = {np.mean(r2_score_Huang):.4f} ± {np.std(r2_score_Huang):.4f}')

