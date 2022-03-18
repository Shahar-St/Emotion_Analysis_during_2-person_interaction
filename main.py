import os

from features_extractor.FeaturesReducer import FeaturesReducer
from features_extractor.RawFeaturesExtractor import RawFeaturesExtractor
from model.Model import Model


def main():
    # Features
    csv_files_dir = ''  # Todo csv or excel?
    csv_files_path = os.path.join(os.getcwd(), csv_files_dir)

    model_train_data = []
    for csv_file in os.listdir(csv_files_path):
        raw_features = RawFeaturesExtractor.extract_raw_features(os.path.join(csv_files_path, csv_file))
        model_Features = FeaturesReducer.process_model_features(raw_features)
        model_train_data.append(model_Features)

    # Model
    model = Model()
    model.train(model_train_data)


if __name__ == '__main__':
    main()
