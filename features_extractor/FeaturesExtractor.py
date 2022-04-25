import os

import numpy as np
import pandas as pd

from features_extractor.Features import *


class FeaturesExtractor:

    def extract_features(self, CSVs_dir_path):
        train_data = []
        train_labels = []
        csv_HRSD_path = os.path.join(CSVs_dir_path, 'HRSD-example-fabricated-data.csv')
        for csv_file_name in os.listdir(CSVs_dir_path):
            # for i in range(10): # replace with for until we have actual input
            # csv_file = ''
            if csv_file_name == '221 w4s4 Video_example 2 1_21_2022 5_43_05 PM 2.csv':
                raw_features_dict, label = FeaturesExtractor.extract_raw_features(os.path.join(CSVs_dir_path,
                                                                                               csv_file_name),
                                                                                  csv_file_name, csv_HRSD_path)
                model_Features = self.process_model_features(raw_features_dict)
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

    def process_model_features(self, raw_features: dict) -> np.array:
        """
        Gets a dict of all raw features and reduce them to the actual model features
        :param: raw_features: dict of the raw features. Example:
            raw_features = {
                'pose_Tx': [4],
                'pose_Ty': [4.5],
                'pose_Rx': [12.7],
            }
        :return: model_features
        """

        features_mapper = {
            pose_stds: ([pose_Tx, pose_Ty, pose_Rx, pose_Ry], lambda x: np.std(x, axis=1)),
        }

        model_features = []
        for feature_name, feature_args in features_mapper.items():
            args = []
            for raw_feature in feature_args[0]:
                args.append(raw_features[raw_feature])
            model_features.extend(feature_args[1](args))

        aus_features = self.extract_features_for_all_aus([
            (raw_features[AU1_binary], raw_features[AU1_intensity]),
            (raw_features[AU2_binary], raw_features[AU2_intensity]),
        ])

        return np.array(model_features + aus_features)

    @staticmethod
    def func1(args):
        a = args
        # logic of the feature
        return [np.max(a)]

    def extract_features_for_all_aus(self, aus_pairs):
        features = []
        for au in aus_pairs:
            features.extend(self.extract_aus_features(au[0], au[1]))
        return features

    @staticmethod
    def extract_aus_features(au_binary_list, au_intensity_list):
        # au_binary_list, au_intensity_list = args
        min_num_of_ones = 5
        max_consecutive_zeros = 2
        max_percentage_of_zeros = 0.5

        phases_indices = []
        counter = 0
        while counter < len(au_binary_list) - min_num_of_ones:
            # find the beginning of a phase
            if sum(au_binary_list[counter:counter + min_num_of_ones]) == min_num_of_ones:
                start_ind = counter
                num_of_ones = min_num_of_ones
                total_num_of_zeros = 0
                num_of_consecutive_zeros = 0
                phase_len = min_num_of_ones

                is_phase_valid = True
                counter += min_num_of_ones
                while is_phase_valid and counter < len(au_binary_list):
                    if au_binary_list[counter] == 1:
                        num_of_consecutive_zeros = 0
                        num_of_ones += 1
                        phase_len += 1
                        counter += 1
                    elif num_of_consecutive_zeros + 1 <= max_consecutive_zeros and (
                            total_num_of_zeros + 1) / phase_len <= max_percentage_of_zeros:
                        num_of_consecutive_zeros += 1
                        total_num_of_zeros += 1
                        phase_len += 1
                        counter += 1
                    else:
                        # phase has ended
                        phases_indices.append((start_ind, counter - 1))
                        is_phase_valid = False

            counter += 1

        # calculate features
        phases_lens = []
        phases_avg_intensities = []
        for start_ind, end_ind in phases_indices:
            phases_lens.append(end_ind - start_ind + 1)
            phases_avg_intensities.append(np.mean(au_intensity_list[start_ind:end_ind + 1]))

        len_avg = np.mean(phases_lens)
        len_std = np.std(phases_lens)
        intensity_avg = np.average(phases_avg_intensities, weights=phases_lens)
        intensity_std = np.std(phases_avg_intensities)

        return [len_avg, len_std, intensity_avg, intensity_std]
