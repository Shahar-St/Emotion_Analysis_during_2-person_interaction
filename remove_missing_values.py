import os.path

import pandas as pd


def remove_missing_values():
    path = os.path.join(os.getcwd(), "database", "db_v0_1.csv")
    # load database v0.1 (with 60 features and after data completion)
    data = get_db(path)
    # remove subjects with missing values
    data.drop(data[data.Subject.isin([557, 413, 102, 109, 112, 218])].index, inplace=True)

    # split database to clinical and sub_clinical
    clinical_df = data[data.Sample == 2]
    sub_clinical_df = data[data.Sample == 1]

    # handle missing values in clinical subjects:
    # features to omit from clinical subjects db
    clinical_features_to_remove = ["BDI_SCORE", "STAI_SCORE", "PSWQ_SCORE", "RRS_SCORE",
                                   "DASS_Depression", "DASS_Anxiety", "DASS_stress",
                                   "ExpectancyConfidance_positive.reject", "ExpectancyRT_positive.reject",
                                   "ExpectancyRejection", "ExpectancyConfidance_negative.reject",
                                   "ExpectancyConfidance_negative.accept", "Subject"]
    clinical_df.drop(clinical_features_to_remove, inplace=True, axis=1)
    # save as csv file

    # handle missing values in sub-clinical subjects:
    # features to omit from sub clinical subjects db
    sub_clinical_features_to_remove = ["BDI_SCORE", "STAI_SCORE", "PSWQ_SCORE", "RRS_SCORE",
                                       "DASS_Depression", "DASS_Anxiety", "DASS_stress",
                                       "ExpectancyConfidance_positive.reject", "ExpectancyAcceptance",
                                       "ExpectancyConfidance_negative.accept",
                                       "ExpectancyConfidance_negative.reject",
                                       "ExpectancyRT_positive.reject", "ExpectancyRejection", "Subject"]
    sub_clinical_df.drop(sub_clinical_features_to_remove, inplace=True, axis=1)

    # handle missing values in db with all subjects:
    # features to omit from db
    combined_features_to_remove = ["BDI_SCORE", "STAI_SCORE", "PSWQ_SCORE", "RRS_SCORE",
                                   "DASS_Depression", "DASS_Anxiety", "DASS_stress",
                                   "ExpectancyConfidance_positive.reject", "ExpectancyAcceptance",
                                   "ExpectancyConfidance_negative.accept",
                                   "ExpectancyConfidance_negative.reject",
                                   "ExpectancyRT_positive.reject", "ExpectancyRejection", "Subject"]
    data.drop(combined_features_to_remove, inplace=True, axis=1)

    # remove rows with blank values
    reomve_Nan_rows(data)
    reomve_Nan_rows(clinical_df)
    reomve_Nan_rows(sub_clinical_df)

    # save as csv file
    sub_clinical_df.to_csv("sub_clinical.csv")
    clinical_df.to_csv("clinical.csv")
    data.to_csv("clinical_and_sub_clinical.csv")


#########################################################################


# remove rows with "NaN" values from the dataframe
def reomve_Nan_rows(data):
    # var for removing "NaN" values
    nan_value = float("NaN")
    data.replace("", nan_value, inplace=True)
    data.dropna(inplace=True)


#########################################################################


# reading the database and saving it into a dataframe
# in[path]: relative database location
# out[df]: database saved as dataframe
def get_db(path):
    # read database file
    # path = "database/db_v6.csv"
    df = pd.read_csv(path)
    return df


def main():
    remove_missing_values()


if __name__ == '__main__':
    main()
