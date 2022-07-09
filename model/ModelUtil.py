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
    def plot_importance_top_k(model, features_names, num_of_features, model_name=None, top=True):
        importance = model.feature_importances_
        if top is True:
            indices = np.argsort(importance)[-num_of_features:]
        else:
            indices = np.argsort(importance)[:num_of_features]
        top_importance = importance[indices]
        top_features_names = features_names[indices]
        std = np.std([tree.feature_importances_[indices] for tree in model.estimators_], axis=0)

        sort_indices = np.argsort(top_importance)
        sorted_top_importance = top_importance[sort_indices]
        sorted_top_features_names = top_features_names[sort_indices]
        sorted_std = std[sort_indices]

        forest_importance = pd.Series(sorted_top_importance, index=sorted_top_features_names)
        fig, ax = plt.subplots()
        forest_importance.plot.barh(yerr=sorted_std, ax=ax)
        suffix = 'Most Important' if top else 'Least Important'
        ax.set_title(f"Feature importance using MDI - {suffix}")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.xticks(fontsize=10)
        if model_name is not None:
            plt.savefig(f'{model_name}-model-top features')
        else:
            plt.show()
        plt.clf()

    @staticmethod
    def plot_comparator_graph(model, data, features_names, num_of_features):
        importance = model.feature_importances_
        indices = np.argpartition(importance, -num_of_features)[-num_of_features:]
        top_importance = importance[indices]
        top_features_names = features_names[indices]
        sort_indices = np.argsort(top_importance)
        sorted_top_features_names = top_features_names[sort_indices][::-1]

        labels_dict = {
            'Anxiety': 'STAI_SCORE',
            'Depression': 'BDI_SCORE'
        }
        for label, label_col in labels_dict.items():
            label_values = ModelUtil.get_values_of_col_name(data, features_names, label_col)
            for feature_num, feature_name in enumerate(sorted_top_features_names):
                feature_data_values = ModelUtil.get_values_of_col_name(data, features_names, feature_name)
                samples = ModelUtil.get_values_of_col_name(data, features_names, 'Sample')
                groups = ModelUtil.get_values_of_col_name(data, features_names, 'Group')

                clinical_sym = []
                clinical_sym_x = []

                clinical_a_sym = []
                clinical_a_sym_x = []

                sub_clinical_sym = []
                sub_clinical_sym_x = []

                sub_clinical_a_sym = []
                sub_clinical_a_sym_x = []

                for feature_val, sample, group, label_value in zip(feature_data_values, samples, groups, label_values):
                    if label == 'Anxiety':
                        if sample == 1 and group in (1, 3):
                            sub_clinical_sym.append(feature_val)
                            sub_clinical_sym_x.append(label_value)
                        if sample == 1 and group not in (1, 3):
                            sub_clinical_a_sym.append(feature_val)
                            sub_clinical_a_sym_x.append(label_value)
                        if sample == 2 and group in (1, 3):
                            clinical_sym.append(feature_val)
                            clinical_sym_x.append(label_value)
                        if sample == 2 and group not in (1, 3):
                            clinical_a_sym.append(feature_val)
                            clinical_a_sym_x.append(label_value)
                    else:
                        if sample == 1 and group in (2, 3):
                            sub_clinical_sym.append(feature_val)
                            sub_clinical_sym_x.append(label_value)
                        if sample == 1 and group not in (2, 3):
                            sub_clinical_a_sym.append(feature_val)
                            sub_clinical_a_sym_x.append(label_value)
                        if sample == 2 and group in (2, 3):
                            clinical_sym.append(feature_val)
                            clinical_sym_x.append(label_value)
                        if sample == 2 and group not in (2, 3):
                            clinical_a_sym.append(feature_val)
                            clinical_a_sym_x.append(label_value)

                marker_size = 35
                marker = '*'
                plt.scatter(clinical_sym_x, clinical_sym, color='red', label='cli_sym', s=marker_size, marker=marker)
                plt.scatter(clinical_a_sym_x, clinical_a_sym, color='orange', label='cli_asym', s=marker_size,
                            marker=marker)
                plt.scatter(sub_clinical_sym_x, sub_clinical_sym, color='blue', label='sub_cli_sym', s=marker_size,
                            marker=marker)
                plt.scatter(sub_clinical_a_sym_x, sub_clinical_a_sym, color='skyblue', label='sub_cli_asym',
                            s=marker_size, marker=marker)

                plt.title(f'{label} | Feature #{feature_num + 1}: {feature_name}')
                plt.xlabel(f'{label_col} score')
                plt.ylabel('feature values')
                plt.legend()
                plt.savefig(f'{label}-Feature #{feature_num + 1}-{feature_name}.png')
                plt.clf()
                # plt.show()

    @staticmethod
    def get_values_of_col_name(data, features_names, col_name):
        index_of_col = np.where(features_names == col_name)[0][0]
        return data[:, index_of_col]

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
