import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Charger les données
spotify_data = pd.read_csv("spotify_data.csv")
#sampled_spotify_data = spotify_data.sample(frac=0.1, random_state=42)  # Réduire la taille de l'échantillon si nécessaire
spotify_data.drop('Unnamed: 0', axis=1, inplace=True)
X = spotify_data.drop('popularity', axis=1)
y = spotify_data['popularity']

# Préparation des données pour la baseline (pas de prétraitement complexe)
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = ['genre']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(), cat_features)
    ])
X_preprocessed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, random_state=42)

# Configuration et entraînement du modèle de baseline (réseau de neurones sans prétraitement avancé)
mlp_baseline = MLPRegressor(hidden_layer_sizes=(100), activation='relu', max_iter=1000, random_state=42)
mlp_baseline.fit(X_train, y_train)  # Entraînement du modèle

# Prédiction et évaluation avec le modèle de baseline
y_test_pred = mlp_baseline.predict(X_test)
y_train_pred = mlp_baseline.predict(X_train)
#==================================================================
mse_test = mean_squared_error(y_test, y_test_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
#==================================================================
rmse_test = np.sqrt(mse_test)
rmse_train = np.sqrt(mse_train)
#==================================================================
mae_test = mean_absolute_error(y_test, y_test_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
#==================================================================
r_squared_test = r2_score(y_test, y_test_pred)
r_squared_train = r2_score(y_train, y_train_pred)
#==================================================================
mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100

# Affichage des résultats
print("MODÈLE DE BASELINE AVEC RÉSEAU DE NEURONES")
print(f"MSE Train: {mse_train:.3f}")
print(f"MSE Test: {mse_test:.3f}")
print(f"-----------------------------------")
print(f"RMSE Train: {rmse_train:.3f}")
print(f"RMSE Test: {rmse_test:.3f}")
print(f"-----------------------------------")
print(f"MAE Train: {mae_train:.3f}")
print(f"MAE Test: {mae_test:.3f}")
print(f"-----------------------------------")
print(f"R-squared Train: {r_squared_train:.3f}")
print(f"R-squared Test: {r_squared_test:.3f}")
print(f"-----------------------------------")
print(f"MAPE Train: {mape_train:.3f}")
print(f"MAPE Test: {mape_test:.3f}")
