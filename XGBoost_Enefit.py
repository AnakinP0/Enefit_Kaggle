
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

try:
    train_path = r'C:\Users\99899\Desktop\Enefit data\train.csv'
    train_df = pd.read_csv(train_path, parse_dates=['datetime'])

    train_df['year'] = train_df['datetime'].dt.year
    train_df['month'] = train_df['datetime'].dt.month
    train_df['day'] = train_df['datetime'].dt.day
    train_df['weekday'] = train_df['datetime'].dt.weekday
    train_df['hour'] = train_df['datetime'].dt.hour

    train_df = train_df.drop(columns=['datetime'])
    clean_df = train_df.dropna(subset=['target'])

    categorical_cols = clean_df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            clean_df[col] = clean_df[col].astype('category')

    x = clean_df.drop(columns='target')
    y = clean_df['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=8)


    if len(categorical_cols) > 0:
        estimators = [
            ('encoder', TargetEncoder()),
            ('clf', XGBRegressor(n_estimators=10, max_depth=3, n_jobs=-1, tree_method='exact'))
        ]
    else:
        estimators = [
            ('clf', XGBRegressor(n_estimators=10, max_depth=3, n_jobs=-1, tree_method='exact'))
        ]

    pipe = Pipeline(steps=estimators)

    search_space = {
    'clf__max_depth' : Integer(6, 24),
    'clf__learning_rate': Real(0.001, 1.0, prior = 'log-uniform'),
    'clf__subsample': Real(0.5, 1.0),
    'clf__colsample_bytree': Real(0.5, 1.0),
    'clf__colsample_bylevel': Real(0.5, 1.0),
    'clf__colsample_bynode': Real (0.5, 1.0),
    'clf__reg_alpha': Real(0.0, 10.0),
    'clf__reg_lambda': Real(0.0, 10.0),
    'clf__gamma': Real(0.0, 10.0)
    }

    opt = BayesSearchCV(pipe, search_space, cv=3, n_iter=20, scoring='neg_mean_absolute_error', random_state=8)

    logging.info("Starting model training...")

    gc.collect()

    opt.fit(x_train, y_train)

    logging.info("Model training completed.")

    score = opt.score(x_test, y_test)
    logging.info(f"Model evaluation score on test set: {score}")

    joblib.dump(opt.best_estimator_, 'trained_model.pkl')

except Exception as e:
    logging.error(f"An error occurred: {e}")







