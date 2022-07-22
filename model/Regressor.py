import logging
import os.path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

from model.ModelUtil import ModelUtil


class Regressor:
    def __init__(self, regressor=None):
        self.random_forest_model = regressor
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(self, features, labels, features_names, model_name, label_name='TEST', show_plots=True, out_dir='output'):
        out_path = os.path.join(os.getcwd(), out_dir)
        log_file_name = f'{model_name}_logs.txt'
        log_file_path = os.path.join(out_path, log_file_name)
        model_path = os.path.join(out_path, f'{model_name}_model')
        if show_plots:
            plots_path = None
        else:
            plots_path = out_path

        open(log_file_path, 'w').close()  # erase existing log file content
        logging.basicConfig(level=logging.INFO, filename=log_file_path, filemode='a+',
                            format='%(message)s')

        logging.info('------------------------Starting Training------------------------')
        logging.info(f'Num of features: {len(features_names)}')
        fold_num = 1
        num_of_folds = 5
        MAEs = []
        MSEs = []
        best_MAE = np.inf
        kf = KFold(n_splits=num_of_folds)
        for train_index, test_index in kf.split(features):
            logging.info(f'\nFold #{fold_num}')

            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            fold_num += 1

            #### Train phase ####
            parameters_to_tune = {
                # 'n_estimators': range(23, 60),
                # 'criterion': ['squared_error', 'absolute_error', 'poisson'],
                # 'max_depth': list(range(20, 40)),
                # 'min_samples_split': list(range(5, 25)),
                # 'min_samples_leaf': list(range(1, 15)),
                # 'max_features': [None, 'sqrt', 'log2'],
                # 'max_leaf_nodes': [None] + list(range(15, 30)),
                # 'oob_score': [True, False],
                # 'ccp_alpha': [0.0, 0.1, 0.2, 0.3, 0.4]
            }

            model = RandomForestRegressor(
                n_jobs=5,
                n_estimators=30,
                criterion='squared_error',
                max_features=None,
                oob_score=True,
                max_depth=5,
                min_samples_split=9
            )

            model = ModelUtil.tune_hyper_params(model, parameters_to_tune, X_train, y_train)

            #### Test phase ####
            preds = np.stack([t.predict(X_test) for t in model.estimators_])
            y_pred = np.mean(preds, axis=0)

            mean_abs_error = mean_absolute_error(y_test, y_pred)
            mean_squ_error = mean_squared_error(y_test, y_pred)
            MAEs.append(mean_abs_error)
            MSEs.append(mean_squ_error)
            logging.info(f'MAE: {mean_abs_error:.2f}')
            logging.info(f'MSE: {mean_squ_error:.2f}')

            if mean_abs_error < best_MAE:
                # save data
                self.X_train = X_train
                self.X_test = X_test
                self.y_train = y_train
                self.y_test = y_test
                self.random_forest_model = model
                self.feature_names = features_names
                joblib.dump(model, model_path)

        # get data from the best model found
        logging.info('\n---Model params---')
        for key, value in self.random_forest_model.get_params().items():
            logging.info(f'{key} : {value}')

        preds = np.stack([t.predict(self.X_test) for t in self.random_forest_model.estimators_])
        y_pred = np.mean(preds, axis=0)
        confidence = np.std(preds, axis=0)

        # sort test and predictions by confidence (std)
        indices = confidence.argsort()
        y_pred = y_pred[indices]
        confidence = confidence[indices]
        y_test = self.y_test[indices]

        logging.info('\n---TEST vs Predictions sorted (asc) by confidence (std)---')
        logging.info('Test\tPrediction\tConfidence')
        for test, pred, conf in zip(y_test, y_pred, confidence):
            logging.info(f'{test}\t\t{pred:.2f}\t\t{conf:.2f}')

        # sort test and predictions by the label value
        test_ids = np.argsort(y_test)
        test_sorted = y_test[test_ids]
        preds_sorted = y_pred[test_ids]
        errors_sorted = np.abs(test_sorted - preds_sorted)

        # plot test score vs error
        ModelUtil.plot_test_vs_error(test_sorted, errors_sorted, label_name, plots_path)

        # most important features
        ModelUtil.plot_importance_top_k(self.random_forest_model, features_names, 10, plots_path)

        mean_MAE = np.mean(MAEs)
        std_MAE = np.std(MAEs)
        mean_MSE = np.mean(MSEs)
        std_MSE = np.std(MSEs)

        logging.info('\n-----------Final Results-----------')
        logging.info(f'All MAEs: {MAEs}')
        logging.info(f'All MSEs: {MSEs}')
        logging.info(f'MAEs mean: {mean_MAE:.2f}')
        logging.info(f'MAEs std {std_MAE:.2f}')
        logging.info(f'MSEs mean: {mean_MSE:.2f}')
        logging.info(f'MSEs std {std_MSE:.2f}')
