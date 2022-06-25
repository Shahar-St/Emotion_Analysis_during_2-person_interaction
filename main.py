import logging
import os
import time

import numpy as np

from features_extractor.FeaturesExtractor import FeaturesExtractor
from model.Model import Model


def main():
    log_file_name = 'logs_file.txt'
    # open(log_file_name, 'w').close()  # erase existing log file content
    logging.basicConfig(level=logging.INFO, filename=log_file_name, filemode='a+',
                        format='%(message)s')
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    logging.info(f'\n\nNEW TRAIN - Time: {curr_time}\n')
    print(f'Program started, writing logs to {log_file_name}')

    # Features
    data_file_name = 'clinical_and_sub_clinical.csv'
    data_file_dir_name = 'input_files'
    data_path = os.path.join(os.getcwd(), data_file_dir_name, data_file_name)
    train_features, train_labels, features_names = FeaturesExtractor().extract_features(data_path)

    # 123 vs 4 model
    train_features_123vs4, train_labels_123vs4, features_names_123vs4 = \
        adjust_data_to_123_vs_4(train_features, train_labels, features_names)

    model_123_vs_4 = Model()
    output_file_path = os.path.join(os.getcwd(), 'model/model_file_123_vs_4')
    model_123_vs_4.train(train_features_123vs4, train_labels_123vs4, features_names_123vs4, output_file_path)

    print('Program finished')


def adjust_data_to_123_vs_4(train_features, train_labels, features_names):
    new_features = []
    new_labels = []
    # quick and dirty: groups 1/2/3 (clinical and sub) vs 4
    for features, label in zip(train_features, train_labels):
        new_features.append(features.tolist() + [label])
        new_labels.append(1 if label in (1, 2, 3) else 4)

    new_features_names = features_names.tolist() + ['Original_label']

    return np.array(new_features), np.array(new_labels), np.array(new_features_names)


if __name__ == '__main__':
    main()
