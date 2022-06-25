import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV


class ModelUtil:

    @staticmethod
    def plot_importance(model, features_names):
        importance = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
        forest_importance = pd.Series(importance, index=features_names)
        fig, ax = plt.subplots()
        forest_importance.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importance using mean decrease in impurity")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.xticks(fontsize=10)
        plt.show()

    @staticmethod
    def tune_hyper_params(model, parameters_to_tune, features, labels):
        grid_search = GridSearchCV(model, parameters_to_tune, n_jobs=5, cv=5, verbose=1)
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

        return grid_search.best_estimator_
