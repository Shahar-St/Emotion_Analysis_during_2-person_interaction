import logging

import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from model.ModelUtil import ModelUtil


class Classifier:

    def __init__(self, classifier=None):
        self.random_forest_model = classifier
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.acc = None

    def train(self, features, labels, features_names, output_file_path):
        logging.info('------------------------Starting Training------------------------')

        test_size = 0.1
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size,
                                                            shuffle=True)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        #### Train phase ####
        classifier = RandomForestClassifier(
            n_jobs=5,
            # n_estimators=29,
            # criterion='entropy',
            # min_samples_split=7,
        )

        parameters_to_tune = {
            'n_estimators': range(5, 35),  # done
            # 'criterion': ['gini', 'entropy'], # done
            # 'min_samples_split': list(range(4, 11)),  # done
            # 'max_features': [None, 'sqrt', 'log2'],  # done
            # 'ccp_alpha': [0.0, 0.1, 0.2] # done
        }

        classifier = ModelUtil.tune_hyper_params(classifier, parameters_to_tune, X_train, y_train)

        logging.info('\nModel params:')
        for key, value in classifier.get_params().items():
            logging.info(f'{key} : {value}')

        logging.info('')

        #### Test phase ####
        y_pred = classifier.predict(X_test)
        correct = 0
        wrong_from_clinical = 0
        wrong_from_sub_clinical = 0
        for i, (pred, test) in enumerate(zip(y_pred, y_test)):
            if pred == test:
                correct += 1
            else:
                if X_test[i][0] == 2:
                    wrong_from_clinical += 1
                elif X_test[i][0] == 1:
                    wrong_from_sub_clinical += 1
                else:
                    raise RuntimeError('Got sample != 1/2')
        acc = correct / len(y_pred)
        self.acc = acc
        total_wrong = wrong_from_clinical + wrong_from_sub_clinical
        print(f'Classifier Accuracy {acc * 100}%')
        logging.info(f'Accuracy: {acc * 100}%')
        logging.info(f'Wrong from clinical: {(wrong_from_clinical / total_wrong) * 100}%')
        logging.info(f'Wrong from sub-clinical: {(wrong_from_sub_clinical / total_wrong) * 100}%')

        # Plot test results
        plt.title('Predictions vs Test data')
        plt.scatter(range(y_pred.size), y_pred, color="red", label='Test Predictions', s=24)
        plt.scatter(range(y_test.size), y_test, color="blue", label='Test data', facecolors='none')
        plt.legend()
        plt.show()

        # most important features
        ModelUtil.plot_importance_top_k(classifier, features_names, 10)

        # save data
        self.random_forest_model = classifier
        self.feature_names = features_names
        joblib.dump(self, output_file_path)

    @staticmethod
    def init_from_file_name(file_name):
        model: Classifier = joblib.load(file_name)
        return model

    def plot_feature_importance(self, feature_names):
        ModelUtil.plot_importance_top_k(self.random_forest_model, feature_names, 10)
