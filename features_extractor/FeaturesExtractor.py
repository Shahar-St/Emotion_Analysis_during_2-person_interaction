import csv

import numpy as np


class FeaturesExtractor:

    @staticmethod
    def extract_features(data_path, with_label=False, label_col='Group', cols_to_remove=None):
        """
        :param data_path: path to csv file
        :param with_label: whether to leave in the label col
        :param label_col: name of label col
        :param cols_to_remove: list of cols to remove from the data
        :return:
        1. array of arrays (features values) F
        2. array of ints (labels) L (group col)
        3. array of string (features name)
        L[i] is the label associated with the features from F[i]
        """

        if cols_to_remove is None:
            cols_to_remove = []
        with open(data_path) as data:
            all_data, features_names = FeaturesExtractor.get_raw_data(data)

            # get labels
            index_of_label = np.where(features_names == label_col)[0][0]
            train_labels = all_data[:, index_of_label]

            if not with_label:
                all_data, features_names = FeaturesExtractor.remove_col_of_name(all_data, features_names, label_col)

            for col in cols_to_remove:
                all_data, features_names = FeaturesExtractor.remove_col_of_name(all_data, features_names, col)

            train_data = np.array(all_data, dtype=float)
            train_labels = np.array(train_labels, dtype=float).astype(int)

            return train_data, train_labels, np.array(features_names)

    @staticmethod
    def get_data_only_cols(data_path, cols_to_take, label_name=None):
        """
        :param data_path: path to csv file
        :param cols_to_take: all cols to take (need to include the label col as well)
        :param label_name: name of label col. if None, return data and feature names (wo the labels array)
        :return:
        1. array of arrays (features values) F
        2. array of ints (labels) L (group col) - Only if :param:label_name is not None
        3. array of string (features name)
        L[i] is the label associated with the features from F[i]
        """
        with open(data_path) as data:
            all_data, features_names = FeaturesExtractor.get_raw_data(data)

            temp_f_names = features_names.copy()
            for col in temp_f_names:
                if col not in cols_to_take:
                    all_data, features_names = FeaturesExtractor.remove_col_of_name(all_data, features_names, col)

            data = np.array(all_data, dtype=float)
            features_names = np.array(features_names)

            if label_name is not None:
                index_of_label = np.where(features_names == label_name)[0][0]
                train_labels = data[:, index_of_label]
                data, features_names = FeaturesExtractor.remove_col_of_name(all_data, features_names, label_name)
                return data, train_labels, features_names

            return data, features_names

    # Util methods
    @staticmethod
    def remove_col_of_name(data, features_names, col_name):
        indices_to_keep = list(range(data.shape[1]))
        try:
            index_of_col = np.where(features_names == col_name)[0][0]
            indices_to_keep.pop(index_of_col)
            features_names = np.delete(features_names, index_of_col)
            data = data[:, indices_to_keep]
            return data, features_names
        except IndexError as err:
            print(f"Couldn't find col: {col_name}")
            raise err

    @staticmethod
    def get_raw_data(data):
        reader = csv.reader(data)
        all_data = np.array([np.array(line) for line in reader])
        features_names = all_data[0]

        # remove empty cols
        empty_cols = np.where(features_names == '')[0]
        cols_to_keep = list(range(all_data.shape[1]))
        for col in empty_cols:
            cols_to_keep.pop(col)
            features_names = np.delete(features_names, col)

        all_data = all_data[:, cols_to_keep]

        all_data = all_data[1:]  # remove header
        return all_data, features_names
