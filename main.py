import os

from features_extractor.FeaturesReducer import FeaturesReducer
from features_extractor.RawFeaturesExtractor import RawFeaturesExtractor
from model.Model import Model


def main():
    # Features
    excel_files_dir = ''  # Todo csv or excel?
    excel_files_path = os.path.join(os.getcwd(), excel_files_dir)
    raw_features = RawFeaturesExtractor.extract_raw_features(excel_files_path)
    model_Features = FeaturesReducer.process_model_features(raw_features)

    # Model
    model = Model()
    model.train(model_Features)


if __name__ == '__main__':
    main()
