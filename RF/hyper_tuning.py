import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV

dataset = pd.read_csv('../data_files/Final_dataset.csv')
Huang_dataset = pd.read_csv('../data_files/Huang & Massa density.csv')

seed = 42

dataset = np.array(dataset)
Huang_dataset = np.array(Huang_dataset)

X = np.load('../RF/feature.npy')
X = np.nan_to_num(np.float32(X))
y = dataset[:,1]

X_Huang_test = np.load('../RF/feature_Huang.npy')
X_Huang_test = np.nan_to_num(np.float32(X_Huang_test))
y_Huang_test = Huang_dataset[:,1]

X_train , X_test , y_train , y_test = train_test_split(X_Huang_test, y_Huang_test, test_size=0.1, random_state=seed)

random_grid = {'n_estimators': [int(x) for x in range(200,2000,200)],
               'max_features': ['sqrt', 'log2'],
               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)]+[None],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True, False]}

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(X_train, y_train)
print(f'best hyperparameters = {rf_random.best_params_}')
best_random = rf_random.best_estimator_

pred = best_random.predict(X_test)
Huang_pred = best_random.predict(X_Huang_test)

print(f'mae_test = {np.abs(pred - y_test).mean()}')
print(f'mae_Huang = {np.abs(Huang_pred - y_Huang_test).mean()}')
