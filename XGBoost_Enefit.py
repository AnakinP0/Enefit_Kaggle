import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import gc
import logging
import joblib

logging.basicConfig(level=logging.INFO)

def load_and_preprocess_data(filepath):
    """
    Load data from CSV and preprocess it by extracting date features and handling missing values.
    """
    df = pd.read_csv(filepath, parse_dates=['datetime'])
    
    # Extract date-related features
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['weekday'] = df['datetime'].dt.weekday
    df['hour'] = df['datetime'].dt.hour
    
    # Drop original datetime column and rows with missing target values
    df = df.drop(columns=['datetime'])
    df = df.dropna(subset=['target'])
    
    # Convert categorical columns to category dtype
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            df[col] = df[col].astype('category')
    
    return df

def split_data(df):
    """
    Split data into training and testing sets.
    """
    x = df.drop(columns='target')
    y = df['target']
    return train_test_split(x, y, test_size=0.2, random_state=8)

def create_pipeline(categorical_cols):
    """
    Create a machine learning pipeline with optional categorical encoding.
    """
    if len(categorical_cols) > 0:
        estimators = [
            ('encoder', TargetEncoder()),
            ('clf', XGBRegressor(n_estimators=10, max_depth=3, n_jobs=-1, tree_method='exact'))
        ]
    else:
        estimators = [
            ('clf', XGBRegressor(n_estimators=10, max_depth=3, n_jobs=-1, tree_method='exact'))
        ]
    return Pipeline(steps=estimators)

def perform_hyperparameter_optimization(pipe, x_train, y_train):
    """
    Perform hyperparameter optimization using Bayesian search.
    """
    search_space = {
        'clf__max_depth': Integer(6, 24),
        'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
        'clf__subsample': Real(0.5, 1.0),
        'clf__colsample_bytree': Real(0.5, 1.0),
        'clf__colsample_bylevel': Real(0.5, 1.0),
        'clf__colsample_bynode': Real(0.5, 1.0),
        'clf__reg_alpha': Real(0.0, 10.0),
        'clf__reg_lambda': Real(0.0, 10.0),
        'clf__gamma': Real(0.0, 10.0)
    }

    opt = BayesSearchCV(pipe, search_space, cv=3, n_iter=20, scoring='neg_mean_absolute_error', random_state=8)
    logging.info("Starting model training...")
    gc.collect()
    opt.fit(x_train, y_train)
    logging.info("Model training completed.")
    return opt

def main():
    try:
        train_path = r'C:\Users\99899\Desktop\Enefit data\train.csv'
        df = load_and_preprocess_data(train_path)
        
        x_train, x_test, y_train, y_test = split_data(df)
        
        pipe = create_pipeline(df.select_dtypes(include=['category']).columns)
        
        opt = perform_hyperparameter_optimization(pipe, x_train, y_train)
        
        score = opt.score(x_test, y_test)
        logging.info(f"Model evaluation score on test set: {score}")
        
        joblib.dump(opt.best_estimator_, 'trained_model.pkl')

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()







