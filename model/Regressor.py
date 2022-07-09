import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

from model.ModelUtil import ModelUtil


class Regressor:
    def __init__(self, regressor=None):
        self.random_forest_model = regressor
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(self, features, labels, features_names, output_file_path):
        logging.info('------------------------Starting Training------------------------')

        test_size = 0.2
        logging.info(f'Test size: {test_size}')
        min_score = labels.min()
        max_score = labels.max()
        logging.info(f'Range = [{min_score}, {max_score}]')
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size,
                                                            shuffle=True)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        #### Train phase ####
        model = RandomForestRegressor(
            n_jobs=5,
            n_estimators=30,
            criterion='absolute_error',
            max_features=None,
            oob_score=True,
            max_depth=62,
            min_samples_split=16
        )

        parameters_to_tune = {
            # 'n_estimators': range(25, 32),
            # 'criterion': ['squared_error', 'absolute_error', 'poisson'],
            # 'max_depth': list(range(59, 65)),  # + [None],
            # 'min_samples_split': list(range(5, 25)),
            # 'min_samples_leaf': list(range(1, 15)),
            # 'max_features': [None, 'sqrt', 'log2'],
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
        preds = np.stack([t.predict(X_test) for t in model.estimators_])
        y_pred = np.mean(preds, axis=0)
        confidence = np.std(preds, axis=0)
        indices = confidence.argsort()
        y_pred = y_pred[indices]
        confidence = confidence[indices]
        y_test = y_test[indices]

        mean_abs_error = mean_absolute_error(y_test, y_pred)
        mean_abs_percentage_error = mean_absolute_percentage_error(y_test, y_pred)
        abs_error_arr = np.absolute(y_pred - y_test)

        print(f'Mean abs error: {mean_abs_error:.2f}')
        print(f'Mean abs percentage error: {(mean_abs_percentage_error * 100):.2f}%')
        logging.info(f'Mean abs error: {mean_abs_error:.2f}')
        logging.info(f'Mean abs percentage error: {(mean_abs_percentage_error * 100):.2f}%')
        logging.info(f'Test errors: {abs_error_arr}')
        logging.info('')
        logging.info(f'Test confidence: {confidence}')

        # Plot test results
        plt.title('Predictions vs Test data sorted (asc) by std')
        x = range(y_pred.size)
        y = [(t, p) for t, p in zip(y_test, y_pred)]
        plt.plot(x, [i for (i, j) in y], 'rs', markersize=4, label='Test data')
        plt.plot(x, [j for (i, j) in y], 'bo', markersize=4, label='Test Predictions')
        plt.plot((x, x), ([i for (i, j) in y], [j for (i, j) in y]), c='black')
        plt.legend()
        plt.show()

        # confidence vs error
        plt.title('Confidence vs test absolute errors')
        plt.plot(confidence, abs_error_arr)
        plt.xlabel('STD')
        plt.ylabel('Error')
        plt.show()

        # most important features
        ModelUtil.plot_importance_top_k(model, features_names, len(features_names))
        ModelUtil.plot_importance_top_k(model, features_names, int(len(features_names) / 2))
        ModelUtil.plot_importance_top_k(model, features_names, int(len(features_names) / 2), top=False)

        # save data
        self.random_forest_model = model
        self.feature_names = features_names
        joblib.dump(model, output_file_path)
