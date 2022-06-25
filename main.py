import logging
import os
import time

import numpy as np

from features_extractor.FeaturesExtractor import FeaturesExtractor
from model.Classifier import Classifier
from model.Regressor import Regressor


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
    train_features, train_labels, features_names = FeaturesExtractor.extract_features(data_path)

    # 123 vs 4 model
    run_model_123_vs_4(train_features, train_labels, features_names)

    run_anxiety_model(train_features, train_labels, features_names)

    run_depression_model(train_features, train_labels, features_names)

    print('Program finished')


def run_model_123_vs_4(train_features, train_labels, features_names):
    # quick and dirty: groups 1/2/3 (clinical and sub) vs 4

    # adjust labels
    new_labels = []
    for label in train_labels:
        new_labels.append(0 if label in (1, 2, 3) else 1)
    train_labels = np.array(new_labels)

    model_123_vs_4 = Classifier()
    output_file_path = os.path.join(os.getcwd(), 'model/model_file_123_vs_4')
    model_123_vs_4.train(train_features, train_labels, features_names, output_file_path)


def run_anxiety_model(train_features, train_labels, features_names):
    pass


def run_depression_model(train_features, train_labels, features_names):
    pass


if __name__ == '__main__':
    main()
