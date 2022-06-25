import logging

import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from model.ModelUtil import ModelUtil


class Regressor:
    def __init__(self):
        self.random_forest_model = None

    def train(self, features, labels, features_names, output_file_path):
        logging.info('------------------------Starting Training------------------------')

        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, stratify=labels)

        #### Train phase ####

        model = RandomForestRegressor(
            n_jobs=5
        )

        parameters_to_tune = {
            'n_estimators': range(23, 30),
            # 'criterion': ['squared_error', 'absolute_error', 'poisson'],
            # 'max_depth': [None] + list(range(5, 15)),
            # 'min_samples_split': list(range(2, 15)),
            # 'min_samples_leaf': list(range(1, 15)),
            # 'max_features': ['auto', 'sqrt', 'log2'],
            # 'max_leaf_nodes': [None] + list(range(15, 30)),
            # 'oob_score': [True, False],
            # 'ccp_alpha': [0.0, 0.1, 0.2, 0.3, 0.4]
        }

        model = ModelUtil.tune_hyper_params(model, parameters_to_tune, X_train, y_train)

        logging.info('\nModel params:')
        for key, value in model.get_params().items():
            logging.info(f'{key} : {value}')

        logging.info('')

        #### Test phase ####
        y_pred = model.predict(X_test)
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
        ModelUtil.plot_importance(model, features_names)

        # save data
        self.random_forest_model = model
        joblib.dump(model, output_file_path)
