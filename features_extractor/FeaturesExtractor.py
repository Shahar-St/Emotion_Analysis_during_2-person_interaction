import csv

import numpy as np


class FeaturesExtractor:

    @staticmethod
    def extract_features(data_path):
        # Get the input file, read the data and extract:
        # 1. array of arrays (features values) F
        # 2. array of ints (labels) L (group col)
        # 3. array of string (features name)
        # L[i] is the label associated with the features from F[i]

        with open(data_path) as data:
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

            # get labels
            index_of_label = np.where(features_names == 'Group')[0][0]
            train_labels = all_data[:, index_of_label]

            train_data, features_names = FeaturesExtractor.remove_col_of_name(all_data, features_names, 'Group')

            train_data = np.array(train_data, dtype=float)
            train_labels = np.array(train_labels, dtype=int)

            return np.array(train_data), np.array(train_labels), np.array(features_names)

    @staticmethod
    def remove_col_of_name(data, features_names, col_name):
        indices_to_keep = list(range(data.shape[1]))
        index_of_col = np.where(features_names == col_name)[0][0]
        indices_to_keep.pop(index_of_col)
        features_names = np.delete(features_names, index_of_col)
        data = data[:, indices_to_keep]
        return data, features_names
