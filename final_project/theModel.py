import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from itertools import combinations

# Load dataset
def load_data(file_path):
    """
    Load dataset from a CSV file.
    
    Parameters:
    file_path (str): Path to the dataset file.
    
    Returns:
    pd.DataFrame: Loaded dataset.
    """
    data = pd.read_csv(file_path)
    return data

# Data Preprocessing
def preprocess_data(df, target_column):
    """
    Preprocesses the dataset by handling missing values, encoding categorical features,
    and normalizing numerical features.
    
    Parameters:
    df (pd.DataFrame): Input dataset.
    target_column (str): Column name of the target variable.
    
    Returns:
    tuple: Preprocessor object, feature matrix (X), and target vector (y).
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    
    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
    
    if df[target_column].isna().sum() > 0:
        df[target_column] = df[target_column].fillna(df[target_column].median())
    
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if target_column not in numeric_features:
        raise ValueError("Target column must be numeric")
    
    numeric_features.remove(target_column)
    
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('power', PowerTransformer(method='yeo-johnson'))
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return preprocessor, X, y

# Feature Selection
def feature_selection(X, y):
    """
    Selects relevant features using Mutual Information, Variance Thresholding,
    and SHAP feature importance analysis.
    
    Parameters:
    X (pd.DataFrame): Feature matrix.
    y (pd.Series): Target variable.
    
    Returns:
    pd.DataFrame: Feature matrix with selected features.
    """
    mi_scores = mutual_info_regression(X, y)
    mi_threshold = np.percentile(mi_scores, 20)
    selected_features = [X.columns[i] for i in range(len(mi_scores)) if mi_scores[i] > mi_threshold]
    X_selected = X[selected_features]
    
    var_thresh = VarianceThreshold(threshold=0.01)
    X_selected = pd.DataFrame(var_thresh.fit_transform(X_selected), columns=X_selected.columns[var_thresh.get_support()])
    
    model = xgb.XGBRegressor()
    model.fit(X_selected, y)
    explainer = shap.Explainer(model)
    shap_values = explainer(X_selected)
    feature_importance = np.abs(shap_values.values).mean(axis=0)
    shap_threshold = np.percentile(feature_importance, 50)
    final_features = [X_selected.columns[i] for i in range(len(feature_importance)) if feature_importance[i] > shap_threshold]
    
    return X_selected[final_features]

# Feature Engineering
def feature_engineering(X):
    """
    Generates new features based on interactions between existing numerical features.
    
    Parameters:
    X (pd.DataFrame): Feature matrix.
    
    Returns:
    pd.DataFrame: Enhanced feature matrix with interaction terms.
    """
    new_features_dict = {}
    
    for col1, col2 in combinations(X.columns, 2):
        new_features_dict[f"{col1}_{col2}_interaction"] = X[col1] * X[col2]
    
    new_features = pd.DataFrame(new_features_dict)
    X_engineered = pd.concat([X, new_features], axis=1).copy()
    
    return X_engineered

# Model Training & Evaluation
def train_evaluate(X, y):
    """
    Splits the data into training and testing sets, trains an XGBoost regression model,
    and evaluates its performance using MSE, RMSE, and R².
    
    Parameters:
    X (pd.DataFrame): Feature matrix.
    y (pd.Series): Target variable.
    
    Returns:
    xgb.XGBRegressor: Trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R² Score: {r2}")
    
    return model

# Main Function
def main(file_path, target_column):
    """
    Executes the full data processing, feature selection, feature engineering,
    and model training pipeline.
    
    Parameters:
    file_path (str): Path to the dataset.
    target_column (str): Name of the target variable.
    
    Returns:
    xgb.XGBRegressor: Trained model.
    """
    df = load_data(file_path)
    preprocessor, X, y = preprocess_data(df, target_column)
    
    X_transformed = pd.DataFrame(preprocessor.fit_transform(X))
    X_transformed.columns = preprocessor.get_feature_names_out()
    
    X_selected = feature_selection(X_transformed, y)
    X_engineered = feature_engineering(X_selected)
    model = train_evaluate(X_engineered, y)
    
    return model

# add the path to the desired dataset and the target column to Run the main function
model = main("data/vehicle.csv", "price")