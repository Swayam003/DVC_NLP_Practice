import argparse
import os
import logging
from src.utils.all_utils import read_yaml, create_directories, get_df
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from src.utils.featurize import save_matrix

STAGE = "Two"

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def main(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    logging.info("Read the yaml file successfully")

    # Reading various config parameters
    artifacts = config["artifacts"]
    prepared_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["PREPARED_DATA"])
    train_data_path = os.path.join(prepared_data_dir_path, artifacts["TRAIN_DATA"]) # path for artifacts/prepared/train.tsv
    test_data_path = os.path.join(prepared_data_dir_path, artifacts["TEST_DATA"]) # path for artifacts/prepared/test.tsv

    featurized_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["FEATURIZED_DATA"])
    create_directories([featurized_data_dir_path]) # Creating directory for artifacts/features

    featurized_train_data_path = os.path.join(featurized_data_dir_path, artifacts["FEATURIZED_OUT_TRAIN"]) # path for artifacts/features/train.pkl
    featurized_test_data_path = os.path.join(featurized_data_dir_path, artifacts["FEATURIZED_OUT_TEST"]) # path for artifacts/features/test.pkl

    max_features = params["featurize"]["max_features"]
    ngrams = params["featurize"]["ngrams"]

    df_train = get_df(train_data_path) # Converting Tab sep file to DataFrame

    train_words = np.array(df_train.text.str.lower().values.astype("U")) # Converting DataFrame to array of LowerCase strings

    # Creating Bag of Words using CountVectorizer
    bag_of_words = CountVectorizer( stop_words ="english", max_features= max_features, ngram_range=(1,ngrams) )
    bag_of_words.fit(train_words)
    train_words_binary_matrix = bag_of_words.transform(train_words)

    # Converting it into Matrix
    tfidf = TfidfTransformer(smooth_idf=False)
    tfidf.fit(train_words_binary_matrix)
    train_words_tfidf_matrix = tfidf.transform(train_words_binary_matrix)

    save_matrix(df_train, train_words_tfidf_matrix, featurized_train_data_path) # Saving Matrix in pickle file

    # Now working to generate featurized test matrix
    df_test = get_df(test_data_path)
    test_words = np.array(df_test.text.str.lower().values.astype("U"))
    test_words_binary_matrix = bag_of_words.transform(test_words)
    test_words_tfidf_matrix = tfidf.transform(test_words_binary_matrix)

    save_matrix(df_test, test_words_tfidf_matrix, featurized_test_data_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage Two started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage Two completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e