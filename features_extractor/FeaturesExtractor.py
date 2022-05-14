import os

import numpy as np
import pandas as pd

raw_features_names = (
    pose_Tx,
    pose_Ty,
    pose_Tz,
    pose_Rx,
    pose_Ry,
    pose_Rz,
    AU1_binary,
    AU1_intensity,
    AU2_binary,
    AU2_intensity
) = ' pose_Tx', \
    ' pose_Ty', \
    ' pose_Tz', \
    ' pose_Rx', \
    ' pose_Ry', \
    ' pose_Rz', \
    ' AU01_c', \
    ' AU01_r', \
    ' AU02_c', \
    ' AU02_r',


class FeaturesExtractor:

    def extract_features(self, CSVs_dir_path, num_of_parts=1, parts_range=None):
        if parts_range is None:
            parts_range = [[0, 1]]
        train_data = []
        train_labels = []
        features_names = []
        csv_HRSD_path = os.path.join(CSVs_dir_path, 'HRSD-example-fabricated-data.csv')
        num_of_sessions = 3
        minutes_per_session = [6, 4, 2]  # 12 minutes in total.
        frames_per_session = [element * 1500 for element in minutes_per_session]  # 1500 frames per minute.
        session_number = 0

        for csv_file_name in os.listdir(CSVs_dir_path):
            # for i in range(10): # replace with for until we have actual input
            # csv_file = ''
            if csv_file_name == '221 w4s4 Video_example 2 1_21_2022 5_43_05 PM 2.csv':
                raw_features_dict, label = FeaturesExtractor.extract_raw_features(os.path.join(CSVs_dir_path,
                                                                                               csv_file_name),
                                                                                  csv_file_name, csv_HRSD_path,
                                                                                  num_of_sessions, frames_per_session,
                                                                                  session_number)
                model_Features, features_names = self.process_model_features(raw_features_dict)
                train_data.append(model_Features)
                train_labels.append(label)

        return np.array(train_data), np.array(train_labels), np.array(features_names)

    @staticmethod
    def extract_raw_features(csv_file_path, csv_file_name, csv_HRSD_path, num_of_sessions, frames_per_session,
                             session_number):
        raw_features_values = []
        raw_features_values_models = []
        csv_file = pd.read_csv(csv_file_path)
        start_session = 0

        for i in range(num_of_sessions):
            end_session = start_session + frames_per_session[i]
            for col in raw_features_names:
                col_val = csv_file[col][start_session: end_session].tolist()
                raw_features_values.append(col_val)

            raw_features_values_models.append(raw_features_values)
            start_session = end_session
            raw_features_values = []

        split_name = csv_file_name.split(" ")
        patient_number = int(split_name[0])
        meeting_number = int(split_name[1].split("s")[1])

        csv_file = pd.read_csv(csv_HRSD_path)
        index_row = 0
        for index in range(len(csv_file['patient'].tolist())):
            if csv_file.iloc[index, 0] == patient_number and csv_file.iloc[index, 1] == meeting_number:
                index_row = index

        label = csv_file.iloc[index_row, 10]

        return dict(zip(raw_features_names, raw_features_values_models[session_number])), label

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

        # list of (model features names, raw features names, function to apply)
        features_mapper = [
            (
                ['pose_Tx_std', 'pose_Ty_std', 'pose_Tz_std', 'pose_Rx_std', 'pose_Ry_std', 'pose_Rz_std'],
                [pose_Tx, pose_Ty, pose_Tz, pose_Rx, pose_Ry, pose_Rz],
                lambda x: np.std(x, axis=1)
            ),
        ]

        model_features = []
        model_features_names_in_order = []
        for model_names, feature_names, func in features_mapper:
            args = [raw_features[raw_feature] for raw_feature in feature_names]
            model_features.extend(func(args))
            model_features_names_in_order.extend(model_names)

        aus_features = self.extract_features_for_all_aus([
            (raw_features[AU1_binary], raw_features[AU1_intensity]),
            (raw_features[AU2_binary], raw_features[AU2_intensity]),
        ])

        aus = [1, 2]
        au_features_names = ['len_avg', 'len_std', 'intensity_avg', 'intensity_std', 'num_of_phases']
        for au in aus:
            for au_features_name in au_features_names:
                model_features_names_in_order.append(f'AU{au}_{au_features_name}')
        model_features_names_in_order.append('Liveliness')

        return np.array(model_features + aus_features), model_features_names_in_order

    def extract_features_for_all_aus(self, aus_pairs):
        features = []
        liveliness = 0
        for au in aus_pairs:
            au_features, new_liveliness = self.extract_aus_features(au[0], au[1])
            features.extend(au_features)
            liveliness += new_liveliness
        features.append(liveliness)
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

        phases_lens = np.array(phases_lens)

        num_of_phases = phases_lens.size
        len_avg = np.mean(phases_lens)
        len_std = np.std(phases_lens)
        intensity_avg = np.average(phases_avg_intensities, weights=phases_lens)
        intensity_std = np.std(phases_avg_intensities)

        au_liveliness = np.sum(phases_lens)

        return [len_avg, len_std, intensity_avg, intensity_std, num_of_phases], au_liveliness
