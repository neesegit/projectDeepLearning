from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, mean_squared_log_error, precision_score, r2_score, mean_squared_error, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
import pandas as pd
import numpy as np

def linearReg(X_train,Y_train,X_test,Y_test):

    linear_regression = LinearRegression().fit(X_train,Y_train)

    linear_prediction = linear_regression.predict(X_test)

    linear_mse = mean_squared_error(Y_test,linear_prediction)
    linear_rmse = np.sqrt(linear_mse)
    linear_mae = mean_absolute_error(Y_test,linear_prediction)
    linear_r2 = r2_score(Y_test,linear_prediction)

    print("ML ALGORITHM")
    print(f"Mean squared error: {linear_mse}")
    print("-"*50)
    print(f"Root mean squared error: {linear_rmse}")
    print("-"*50)
    print(f"Mean absolute error: {linear_mae}")
    print("-"*50)
    print(f"R2 score: {linear_r2}")
    print("="*50)

def dummyReg(X_train, y_train, X_test, y_test):
    
    # Initialize the Dummy Regressor
    dummy_regressor = DummyRegressor(strategy="mean")

    # Fit the Dummy Regressor to the training data
    dummy_regressor.fit(X_train, y_train)

    # Make predictions with the Dummy Regressor
    y_train_pred_dummy = dummy_regressor.predict(X_train)
    y_test_pred_dummy = dummy_regressor.predict(X_test)

    # Calculate metrics for Dummy Regressor
    mse_train_dummy = mean_squared_error(y_train, y_train_pred_dummy)
    mse_test_dummy = mean_squared_error(y_test, y_test_pred_dummy)

    rmse_train_dummy = np.sqrt(mse_train_dummy)
    rmse_test_dummy = np.sqrt(mse_test_dummy)

    mae_train_dummy = mean_absolute_error(y_train, y_train_pred_dummy)
    mae_test_dummy = mean_absolute_error(y_test, y_test_pred_dummy)

    r_squared_train_dummy = r2_score(y_train, y_train_pred_dummy)
    r_squared_test_dummy = r2_score(y_test, y_test_pred_dummy)

    msle_train_dummy = mean_squared_log_error(y_train, y_train_pred_dummy)
    msle_test_dummy = mean_squared_log_error(y_test, y_test_pred_dummy)

    mape_train_dummy = np.mean(np.abs((y_train - y_train_pred_dummy) / y_train)) * 100
    mape_test_dummy = np.mean(np.abs((y_test - y_test_pred_dummy) / y_test)) * 100

    # Print metrics for Dummy Regressor
    print("DUMMY REGRESSOR")
    print("-"*30)
    print(f"MSE Train: {mse_train_dummy:.3f}")
    print(f"MSE Test: {mse_test_dummy:.3f}")
    print("-"*30)
    print(f"RMSE Train: {rmse_train_dummy:.3f}")
    print(f"RMSE Test: {rmse_test_dummy:.3f}")
    print("-"*30)
    print(f"MAE Train: {mae_train_dummy:.3f}")
    print(f"MAE Test: {mae_test_dummy:.3f}")
    print("-"*30)
    print(f"R-squared Train: {r_squared_train_dummy:.3f}")
    print(f"R-squared Test: {r_squared_test_dummy:.3f}")
    print("-"*30)
    print(f"MSLE Train: {msle_train_dummy:.3f}")
    print(f"MSLE Test: {msle_test_dummy:.3f}")
    print("-"*30)
    print(f"MAPE Train: {mape_train_dummy:.3f}")
    print(f"MAPE Test: {mape_test_dummy:.3f}")
    print("="*50)


# Load the dataset
spotify_data = pd.read_csv("spotify_data.csv")

# Drop the 'Unnamed: 0' column if present
spotify_data.drop('Unnamed: 0', axis=1, inplace=True, errors='ignore')

# Sample the dataset (e.g., 10% of the data)
sampled_spotify_data = spotify_data.sample(frac=0.1, random_state=42)  # Adjust frac as needed

X = sampled_spotify_data.drop('popularity', axis=1)
y = sampled_spotify_data['popularity']

# Define numerical and categorical features
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = ['genre']

# Preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(), cat_features)
    ])

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

X_prepared = pipeline.fit_transform(X)

# Generate column names for the processed features
column_names = num_features + \
               list(pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(cat_features))

X_prepared_df = pd.DataFrame(X_prepared.toarray(), columns=column_names)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_prepared_df, y, test_size=0.2, random_state=42)

linearReg(X_train, y_train, X_test, y_test)
#dummyReg(X_train, y_train, X_test, y_test)