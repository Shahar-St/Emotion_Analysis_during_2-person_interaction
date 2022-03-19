import os
import logging
import numpy as np

from features_extractor.FeaturesExtractor import FeaturesExtractor
from model.Model import Model


def main():
    log_file_name = 'logs_file.txt'
    # open(log_file_name, 'w').close()  # erase existing log file content
    logging.basicConfig(level=logging.INFO, filename=log_file_name, filemode='a+',
                        format='%(asctime)-15s %(levelname)-8s %(message)s')
    print(f'Program started, writing logs to {log_file_name}')
    # Features
    CSVs_dir_name = 'input_files'  # Todo csv or excel?
    CSVs_dir_path = os.path.join(os.getcwd(), CSVs_dir_name)
    train_data, train_labels = FeaturesExtractor.extract_features(CSVs_dir_path)

    # Model
    model = Model()
    output_file_path = os.path.join(os.getcwd(), 'model/model_file')
    model.train(np.array(train_data), np.array(train_labels), output_file_path)

    print('Program finished')


if __name__ == '__main__':
    main()
