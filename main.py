import os

import numpy as np

from features_extractor.Consts import *
from features_extractor.FeaturesExtractor import FeaturesExtractor
from model.Classifier import Classifier
from model.Regressor import Regressor


def main():
    print('Program Started')
    build_and_train_task_regression()
    print('Program finished')


def extract_data_wo_cols(file_name, with_label=False, label_col='Group', cols_to_remove=None):
    data_file_dir_name = 'input_files'
    data_path = os.path.join(os.getcwd(), data_file_dir_name, file_name)
    train_features, train_labels, features_names = FeaturesExtractor.extract_features(data_path, with_label, label_col,
                                                                                      cols_to_remove)
    return train_features, train_labels, features_names


def extract_data_only_from_cols(file_name, from_cols, label_col='Group'):
    data_file_dir_name = 'input_files'
    data_path = os.path.join(os.getcwd(), data_file_dir_name, file_name)
    train_features, train_labels, features_names = FeaturesExtractor.get_data_only_cols(data_path, from_cols, label_col)

    return train_features, train_labels, features_names


def build_and_train_123_vs_4():
    file_name = 'clinical_and_sub_clinical.csv'
    train_features, train_labels, features_names = extract_data_wo_cols(file_name)
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
    train_features, train_labels, features_names = extract_data_wo_cols(file_name)
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


def build_and_train_regression(cols, to_remove, label_name, model_name, withDuplicatingData, dup_threshold, show_plots,
                               out_dir='output'):
    file_name = 'clinical_and_sub_clinical_with_score.csv'
    if to_remove:
        train_features, train_labels, features_names = extract_data_wo_cols(file_name, label_col=f'{label_name}_SCORE',
                                                                            cols_to_remove=cols)
    else:
        train_features, train_labels, features_names = extract_data_only_from_cols(file_name, cols,
                                                                                   label_col=f'{label_name}_SCORE')
    if withDuplicatingData:
        extra_features = []
        extra_labels = []
        for train_feature, train_label in zip(train_features, train_labels):
            if train_label >= dup_threshold:
                extra_features.append(train_feature)
                extra_labels.append(train_label)
        train_features = np.array(train_features.tolist() + extra_features)
        train_labels = np.array(train_labels.tolist() + extra_labels)

    model = Regressor()
    model.train(train_features, train_labels, features_names, model_name, label_name=label_name, show_plots=show_plots,
                out_dir=out_dir)


def build_and_train_anxiety_regression(withDuplicatingData=False):
    cols_to_remove = [
        'Group',
        'BDI_SCORE',
        'Sample'
    ]
    suffix = '_wd' if withDuplicatingData else ''
    model_name = f'anxiety_regression{suffix}'
    build_and_train_regression(cols_to_remove, True, 'STAI', model_name, withDuplicatingData, 50, show_plots=False)


def build_and_train_depression_regression(withDuplicatingData=False):
    cols_to_remove = [
        'Group',
        'STAI_SCORE',
        'Sample'
    ]
    suffix = '_wd' if withDuplicatingData else ''
    model_name = f'depression_regression{suffix}'
    build_and_train_regression(cols_to_remove, True, 'BDI', model_name, withDuplicatingData, 29, show_plots=False)


def build_and_train_anxiety_regression_wo_Neg(withDuplicatingData=False):
    cols_to_remove = [
        'Group',
        'BDI_SCORE',
        'Sample',
        'Interpretation_NegSelection'
    ]
    suffix = '_wd' if withDuplicatingData else ''
    model_name = f'anxiety_regression_wo_Neg{suffix}'
    build_and_train_regression(cols_to_remove, True, 'STAI', model_name, withDuplicatingData, 50, show_plots=False)


def build_and_train_depression_regression_wo_Neg(withDuplicatingData=False):
    cols_to_remove = [
        'Group',
        'STAI_SCORE',
        'Sample',
        'Interpretation_NegSelection'
    ]
    suffix = '_wd' if withDuplicatingData else ''
    model_name = f'depression_regression{suffix}'
    build_and_train_regression(cols_to_remove, True, 'BDI', model_name, withDuplicatingData, 29, show_plots=False)


def build_and_train_anxiety_regression_implicit(withDuplicatingData=False):
    suffix = '_wd' if withDuplicatingData else ''
    model_name = f'anxiety_regression_implicit{suffix}'
    build_and_train_regression(['STAI_SCORE'] + implicit_features, False, 'STAI', model_name, withDuplicatingData, 50,
                               show_plots=False)


def build_and_train_depression_regression_implicit(withDuplicatingData=False):
    suffix = '_wd' if withDuplicatingData else ''
    model_name = f'depression_regression_implicit{suffix}'
    build_and_train_regression(['BDI_SCORE'] + implicit_features, False, 'BDI', model_name, withDuplicatingData, 29,
                               show_plots=False)


def build_and_train_anxiety_regression_explicit(withDuplicatingData=False):
    suffix = '_wd' if withDuplicatingData else ''
    model_name = f'anxiety_regression_implicit{suffix}'
    build_and_train_regression(['STAI_SCORE'] + explicit_features, False, 'STAI', model_name, withDuplicatingData, 50,
                               show_plots=False)


def build_and_train_depression_regression_explicit(withDuplicatingData=False):
    suffix = '_wd' if withDuplicatingData else ''
    model_name = f'depression_regression_implicit{suffix}'
    build_and_train_regression(['BDI_SCORE'] + explicit_features, False, 'BDI', model_name, withDuplicatingData, 29,
                               show_plots=False)


def build_and_train_task_regression():
    for withDuplication in [True, False]:
        for score_name, threshold in thresholds.items():
            for task, features in tasks_to_features_dict.items():
                suffix = '_wd' if withDuplication else ''
                dir_name = f'{score_name}_{task}{suffix}'
                out_dir = os.path.join('output', dir_name)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                build_and_train_regression(features + [f'{score_name}_SCORE'], False, score_name,
                                           f'{score_name}_{task}', withDuplication, threshold, False, out_dir)


if __name__ == '__main__':
    main()
