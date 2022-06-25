import logging
import os

from features_extractor.FeaturesExtractor import FeaturesExtractor
from model.Model import Model


def main():
    log_file_name = 'logs_file.txt'
    open(log_file_name, 'w').close()  # erase existing log file content
    logging.basicConfig(level=logging.INFO, filename=log_file_name, filemode='a+',
                        format='%(message)s')
    print(f'Program started, writing logs to {log_file_name}')

    # Features
    data_file_name = 'clinical_and_sub_clinical.csv'
    data_file_dir_name = 'input_files'
    data_path = os.path.join(os.getcwd(), data_file_dir_name, data_file_name)
    train_data, train_labels, features_names = FeaturesExtractor().extract_features(data_path)

    # Model
    model = Model()
    output_file_path = os.path.join(os.getcwd(), 'model/model_file')
    model.train(train_data, train_labels, features_names, output_file_path)

    print('Program finished')


if __name__ == '__main__':
    main()
