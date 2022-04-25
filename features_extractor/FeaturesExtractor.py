import os

import numpy as np
import pandas as pd

from features_extractor.Features import *


class FeaturesExtractor:

    @staticmethod
    def extract_features(CSVs_dir_path):
        train_data = []
        train_labels = []
        csv_HRSD_path = os.path.join(CSVs_dir_path, 'HRSD-example-fabricated-data.csv')
        for csv_file_name in os.listdir(CSVs_dir_path):
            # for i in range(10): # replace with for until we have actual input
            # csv_file = ''
            if not csv_file_name == 'HRSD-example-fabricated-data.csv':
                raw_features_dict, label = FeaturesExtractor.extract_raw_features(os.path.join(CSVs_dir_path,
                                                                                               csv_file_name),
                                                                                  csv_file_name, csv_HRSD_path)
                model_Features = FeaturesExtractor.process_model_features(raw_features_dict)
                train_data.append(model_Features)
                train_labels.append(label)

        return np.array(train_data), np.array(train_labels)

    @staticmethod
    def extract_raw_features(csv_file_path, csv_file_name, csv_HRSD_path):

        raw_features_values = []
        csv_file = pd.read_csv(csv_file_path)
        for col in raw_features_names:
            col_val = csv_file[col]
            raw_features_values.append(col_val)

        split_name = csv_file_name.split(" ")
        patient_number = int(split_name[0])
        session_number = int(split_name[1].split("s")[1])

        csv_file = pd.read_csv(csv_HRSD_path)
        index_row = 0
        for index in range(len(csv_file['patient'].tolist())):
            if csv_file.iloc[index, 0] == patient_number and csv_file.iloc[index, 1] == session_number:
                index_row = index

        label = (csv_file.iloc[index_row, 10], csv_file.iloc[index_row, 12])

        return dict(zip(raw_features_names, raw_features_values)), label

    @staticmethod
    def process_model_features(raw_features: dict) -> np.array:
        """
        Gets a dict of all raw features and reduce them to the actual model features
        :param: raw_features: dict of the raw features. Example:
            raw_features = {
                'raw_feature1': [4],
                'raw_feature2': [4.5],
                'raw_feature3': [12.7],
            }
        :return: model_features
        """

        features_mapper = {
            model_feature1: ((raw_feature1, raw_feature2, raw_feature3), FeaturesExtractor.func1),
            model_feature2: ((raw_feature1, raw_feature2), min),
        }

        model_features = []
        for feature_name, feature_args in features_mapper.items():
            args = []
            for raw_feature in feature_args[0]:
                args.append(raw_features[raw_feature])
            model_features.append(feature_args[1](args))

        return np.array(model_features)

    @staticmethod
    def func1(args):
        a, b, c = args
        # logic of the feature
        return max(a, b, c)
