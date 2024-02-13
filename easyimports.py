def imports():
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, mean_squared_log_error, precision_score, r2_score, mean_squared_error, mean_absolute_error, roc_auc_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder