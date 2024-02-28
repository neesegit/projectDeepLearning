import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, mean_squared_log_error, precision_score, r2_score, mean_squared_error, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor

spotify_data = pd.read_csv("spotify_data.csv")
sampled_spotify_data = spotify_data.sample(frac=0.1, random_state=42)  
sampled_spotify_data.drop('Unnamed: 0', axis=1, inplace=True)
X = sampled_spotify_data.drop('popularity', axis=1)
y = sampled_spotify_data['popularity']


num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = ['genre']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(), cat_features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

X_prepared = pipeline.fit_transform(X)

column_names = num_features + \
               list(pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(cat_features))

X_prepared_df = pd.DataFrame(X_prepared.toarray(), columns=column_names)

X_train, X_test, y_train, y_test = train_test_split(X_prepared_df, y, test_size=0.2, random_state=42)

param_grid = {
    'hidden_layer_sizes': [(10,10,10), ],
    'activation': ['logistic', 'relu'],
}

mlp = MLPRegressor(max_iter=1000, random_state=42)

grid_search = GridSearchCV(mlp,param_grid, cv=5, scoring="neg_mean_squared_error")
grid_search.fit(X_train,y_train)
results = grid_search.cv_results_
for mean_score, params in zip(results['mean_test_score'], results['params']):
    mlp.set_params(**params)
    mlp.fit(X_train, y_train)
    y_train_pred = mlp.predict(X_train)
    y_test_pred = mlp.predict(X_test)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)

    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    r_squared_train = r2_score(y_train, y_train_pred)
    r_squared_test = r2_score(y_test, y_test_pred)

    #msle_train = mean_squared_log_error(y_train, y_train_pred)
    #msle_test = mean_squared_log_error(y_test, y_test_pred)

    mape_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
    mape_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

    print("NEURAL NETWORK")
    print(f"Parameters: {params}")
    print("-"*30)
    print(f"MSE Train: {mse_train:.3f}")
    print(f"MSE Test: {mse_test:.3f}")
    print("-"*30)
    print(f"RMSE Train: {rmse_train:.3f}")
    print(f"RMSE Test: {rmse_test:.3f}")
    print("-"*30)
    print(f"MAE Train: {mae_train:.3f}")
    print(f"MAE Test: {mae_test:.3f}")
    print("-"*30)
    print(f"R-squared Train: {r_squared_train:.3f}")
    print(f"R-squared Test: {r_squared_test:.3f}")
    #print("-"*30)
    #print(f"MSLE Train: {msle_train:.3f}")
    #print(f"MSLE Test: {msle_test:.3f}")
    print("-"*30)
    print(f"MAPE Train: {mape_train:.3f}")
    print(f"MAPE Test: {mape_test:.3f}")
    print("="*50)