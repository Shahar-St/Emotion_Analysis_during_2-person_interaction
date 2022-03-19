import logging

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class Model:
    def __init__(self):
        self.random_forest_model = None

    def train(self, features, labels, output_file_path):
        X_train_all, X_test, y_train_all, y_test = train_test_split(features, labels, test_size=0.2)

        #### Train phase ####
        X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2)

        # parameters to tune
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier

        best_params = None
        best_model = None
        best_validation_score = -1
        for nun_of_trees in range(1, 10):

            # The function to measure the quality of a split
            criterion = ('gini', 'entropy')
            for func in criterion:
                # The maximum depth of the tree
                # None = nodes are expanded until all leaves are pure
                # or until all leaves contain less than min_samples_split samples
                max_depth = None

                # The minimum number of samples required to split an internal node
                # int will give the number, float -> ceil(min_samples_split * n_samples)
                for min_samples_split in range(2, 10):

                    clf = RandomForestClassifier(
                        n_estimators=nun_of_trees,
                        criterion=func,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        n_jobs=-1
                    )

                    sample_weight = None
                    clf.fit(X_train, y_train, sample_weight)

                    #### Validation phase ####
                    validation_predictions = clf.predict(X_val)
                    validation_score = (np.sum(validation_predictions == y_val) / y_val.size) * 100
                    params = [nun_of_trees, func, min_samples_split]
                    logging.info(f'Finished training with params: {params}, validation score: {validation_score}%')
                    if validation_score > best_validation_score:
                        best_params = params
                        best_validation_score = validation_score
                        best_model = clf
                        logging.info(f'Found a higher validation score')

        logging.info(f'Training phase ended. Highest validation score: {best_validation_score}, params: {best_params}')

        #### Test phase ####
        predictions = best_model.predict(X_test)
        test_results = (np.sum(predictions == y_test) / y_test.size) * 100
        self.random_forest_model = best_model
        logging.info(f'Test results: {test_results}%')
        joblib.dump(best_model, output_file_path)
