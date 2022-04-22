import os
import numpy as np

from features_extractor.Features import *


class FeaturesExtractor:

    @staticmethod
    def extract_features(CSVs_dir_path):
        train_data = []
        train_labels = []
        for csv_file in os.listdir(CSVs_dir_path):
            # for i in range(10): # rplace with for until we have actual input
            #     csv_file = ''
            raw_features_dict, label = FeaturesExtractor.extract_raw_features(os.path.join(CSVs_dir_path, csv_file))
            model_Features = FeaturesExtractor.process_model_features(raw_features_dict)
            train_data.append(model_Features)
            train_labels.append(label)
        return np.array(train_data), np.array(train_labels)

    @staticmethod
    def extract_raw_features(csv_file_path) -> (dict, int):
        raw_features_values = (4, 5, 0.6)
        label = 0

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

    def extract_features_for_all_aus(self, aus_pairs):
        features = []
        for au in aus_pairs:
            features.extend(self.extract_aus_features(au[0], au[1]))
        return features

    @staticmethod
    def extract_aus_features(au_binary_list, au_intensity_list):
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
