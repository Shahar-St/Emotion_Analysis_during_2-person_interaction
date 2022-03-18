from features_extractor.Features import *


class FeaturesReducer:

    @staticmethod
    def process_model_features(raw_features: dict) -> dict:
        """
        Gets a dict of all raw features and reduce them to the actual model features
        :param raw_features: dict of the raw features. Example:
            raw_features = {
                'raw_feature1': 4,
                'raw_feature2': 4.5,
                'raw_feature3': 12.7,
            }
        :return: model_features
        """

        features_mapper = {
            nodel_feature1: ((raw_feature1, raw_feature2, raw_feature3), FeaturesReducer.func1),
            nodel_feature2: ((raw_feature1, raw_feature2), min),
        }

        model_features = {}
        for feature_name, feature_args in features_mapper.items():
            args = []
            for raw_feature in feature_args[0]:
                args.append(raw_features[raw_feature])
            model_features[feature_name] = feature_args[1](args)

        return model_features

    @staticmethod
    def func1(args):
        a, b, c = args
        # logic of the feature
        return max(a, b, c)
