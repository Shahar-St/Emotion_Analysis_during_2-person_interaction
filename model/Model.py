import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


class Model:
    def __init__(self):
        self.random_forest_model = None

    def train(self, features, labels, features_names, output_file_path):
        logging.info('------------------------Starting Training------------------------')

        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, stratify=labels)
        # todo check stratified

        #### Train phase ####
        regressor = RandomForestRegressor(n_jobs=5)

        parameters_to_tune = {
            'n_estimators': range(4, 20),
            # 'criterion': ['squared_error', 'absolute_error', 'poisson'],
            # 'max_depth': [None] + list(range(5, 15)),
            # 'min_samples_split': list(range(2, 15)),
            # 'min_samples_leaf': list(range(1, 15)),
            # 'max_features': ['auto', 'sqrt', 'log2'],
            # 'max_leaf_nodes': [None] + list(range(15, 30)),
            # 'oob_score': [True, False],
            # 'ccp_alpha': [0.0, 0.1, 0.2, 0.3, 0.4]
        }

        grid_search = GridSearchCV(regressor, parameters_to_tune, n_jobs=5, cv=5, verbose=1)
        grid_search.fit(features, labels)

        cv_results = grid_search.cv_results_
        logging.info('CV scores:')

        for param, values in parameters_to_tune.items():
            logging.info(f'\t{param}:')

            indices = [np.where(cv_results['param_' + param] == value)[0][0] for value in values]
            scores = [cv_results['mean_test_score'][i] for i in indices]
            for value, score in zip(values, scores):
                chosen_msg = ' <-- This was chosen' if grid_search.best_params_[param] == value else ''
                logging.info(f'\t\t{value}: {score:.3f}{chosen_msg}')

        regressor = grid_search.best_estimator_

        logging.info('\nModel params:')
        for key, value in regressor.get_params().items():
            logging.info(f'{key} : {value}')

        logging.info('')

        # Plot train results
        train_pred = regressor.predict(X_train)
        plt.title('Train data and error')
        plt.scatter(range(train_pred.size), train_pred, color="black", label='Train predictions')
        plt.plot(y_train, color="blue", linewidth=3, label='Train data')
        plt.legend()
        plt.show()

        #### Test phase ####
        y_pred = regressor.predict(X_test)
        mean_sq_error = mean_squared_error(y_test, y_pred)
        test_error = abs(y_pred - y_test)
        logging.info(f'Mean squared error: {mean_sq_error:.2f}')
        logging.info(f'Test errors: {test_error}')

        # Plot test results
        plt.title('Predictions vs Test data')
        plt.scatter(range(y_pred.size), y_pred, color="black", label='Test Predictions')
        plt.plot(y_test, color="blue", linewidth=3, label='Test data')
        plt.legend()
        plt.show()

        # most important features
        importance = regressor.feature_importances_
        std = np.std([tree.feature_importances_ for tree in regressor.estimators_], axis=0)
        forest_importance = pd.Series(importance, index=features_names)
        fig, ax = plt.subplots()
        forest_importance.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importance using mean decrease in impurity")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.show()

        # save data
        self.random_forest_model = regressor
        joblib.dump(regressor, output_file_path)
