import os

import numpy as np
import pandas as pd


class FeaturesExtractor:

    def extract_features(self, data_path):
        # Get the input file, read the data and extract:
        # 1. array of arrays (features values) F
        # 2. array of ints (labels) L (group col)
        # 3. array of string (features name)
        # L[i] is the label associated with the features from F[i]

        train_data = []
        train_labels = []
        features_names = []

        # csv_HRSD_path = os.path.join(CSVs_dir_path, 'HRSD-example-fabricated-data.csv')
        # num_of_sessions = 3
        # minutes_per_session = [6, 4, 2]  # 12 minutes in total.
        # frames_per_session = [element * 1500 for element in minutes_per_session]  # 1500 frames per minute.
        # session_number = 0
        #
        # for csv_file_name in os.listdir(CSVs_dir_path):
        #     # for i in range(10): # replace with for until we have actual input
        #     # csv_file = ''
        #     if csv_file_name == '221 w4s4 Video_example 2 1_21_2022 5_43_05 PM 2.csv':
        #         raw_features_dict, label = FeaturesExtractor.extract_raw_features(os.path.join(CSVs_dir_path,
        #                                                                                        csv_file_name),
        #                                                                           csv_file_name, csv_HRSD_path,
        #                                                                           num_of_sessions, frames_per_session,
        #                                                                           session_number)
        #         model_Features, features_names = self.process_model_features(raw_features_dict)
        #         train_data.append(model_Features)
        #         train_labels.append(label)

        return np.array(train_data), np.array(train_labels), np.array(features_names)
    #
    # @staticmethod
    # def extract_raw_features(csv_file_path, csv_file_name, csv_HRSD_path, num_of_sessions, frames_per_session,
    #                          session_number):
    #     raw_features_values = []
    #     raw_features_values_models = []
    #     csv_file = pd.read_csv(csv_file_path)
    #     start_session = 0
    #
    #     for i in range(num_of_sessions):
    #         end_session = start_session + frames_per_session[i]
    #         for col in raw_features_names:
    #             col_val = csv_file[col][start_session: end_session].tolist()
    #             raw_features_values.append(col_val)
    #
    #         raw_features_values_models.append(raw_features_values)
    #         start_session = end_session
    #         raw_features_values = []
    #
    #     split_name = csv_file_name.split(" ")
    #     patient_number = int(split_name[0])
    #     meeting_number = int(split_name[1].split("s")[1])
    #
    #     csv_file = pd.read_csv(csv_HRSD_path)
    #     index_row = 0
    #     for index in range(len(csv_file['patient'].tolist())):
    #         if csv_file.iloc[index, 0] == patient_number and csv_file.iloc[index, 1] == meeting_number:
    #             index_row = index
    #
    #     label = csv_file.iloc[index_row, 10]
    #
    #     return dict(zip(raw_features_names, raw_features_values_models[session_number])), label
