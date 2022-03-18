import os

import numpy as np

from features_extractor.FeaturesReducer import FeaturesReducer
from features_extractor.RawFeaturesExtractor import RawFeaturesExtractor
from model.Model import Model


def main():
    # Features
    csv_files_dir = 'input_files'  # Todo csv or excel?
    csv_files_path = os.path.join(os.getcwd(), csv_files_dir)

    train_data = []
    train_labels = []
    for csv_file in os.listdir(csv_files_path):
        raw_features_dict, label = RawFeaturesExtractor.extract_raw_features(os.path.join(csv_files_path, csv_file))
        model_Features = FeaturesReducer.process_model_features(raw_features_dict)
        train_data.append(model_Features)
        train_labels.append(label)

    # Model
    model = Model()
    output_file_path = os.path.join(os.getcwd(), 'model/model_file')
    model.train(np.array(train_data), np.array(train_labels), output_file_path)


if __name__ == '__main__':
    main()
