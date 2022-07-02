import logging
import os

import numpy as np

from features_extractor.FeaturesExtractor import FeaturesExtractor
from model.Classifier import Classifier
from model.ModelUtil import ModelUtil
from model.Regressor import Regressor


def main():
    log_file_name = 'logs_file.txt'
    # open(log_file_name, 'w').close()  # erase existing log file content
    logging.basicConfig(level=logging.INFO, filename=log_file_name, filemode='a+',
                        format='%(message)s')
    print(f'Program started, writing logs to {log_file_name}')

    # build_and_train_123_vs_4()
    # load_model_and_plot()
    build_and_train_1_vs_2()

    print('Program finished')


def extract_data(file_name):
    data_file_dir_name = 'input_files'
    data_path = os.path.join(os.getcwd(), data_file_dir_name, file_name)
    train_features, train_labels, features_names = FeaturesExtractor.extract_features(data_path)
    return train_features, train_labels, features_names


def build_and_train_123_vs_4():
    file_name = 'clinical_and_sub_clinical.csv'
    train_features, train_labels, features_names = extract_data(file_name)
    # adjust labels
    new_labels = []
    for label in train_labels:
        new_labels.append(0 if label in (1, 2, 3) else 1)
    train_labels = np.array(new_labels)

    model_123_vs_4 = Classifier()
    output_file_path = os.path.join(os.getcwd(), 'model/model_file_123_vs_4')
    model_123_vs_4.train(train_features, train_labels, features_names, output_file_path)


def build_and_train_1_vs_2():
    file_name = 'clinical_and_sub_clinical.csv'
    train_features, train_labels, features_names = extract_data(file_name)
    # adjust data
    new_labels = []
    new_features = []
    for i, label in enumerate(train_labels):
        if label == 1:
            new_labels.append(1)
            new_features.append(train_features[i])
        elif label == 2:
            new_labels.append(0)
            new_features.append(train_features[i])

    train_labels = np.array(new_labels)
    train_features = np.array(new_features)

    model_1_vs_2 = Classifier()
    output_file_path = os.path.join(os.getcwd(), 'model/model_file_1_vs_2')
    model_1_vs_2.train(train_features, train_labels, features_names, output_file_path)


def load_model_and_plot():
    model_path = os.path.join(os.getcwd(), 'model', 'model_file_1_vs_2')
    model = Classifier.init_from_file_name(model_path)
    file_name = 'clinical_and_sub_clinical_with_score.csv'
    train_features, train_labels, features_names = extract_data(file_name)
    model.feature_names = features_names
    ModelUtil.plot_comparator_graph(train_features, features_names)


if __name__ == '__main__':
    main()
