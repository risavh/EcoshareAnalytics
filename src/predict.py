import logging
import os
import pickle
import warnings
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
# import pymssql
import yaml
import json

# from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


def read_config():
    """reads config files and returns input params as a nested dictionary"""
    with open("config.yaml") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        # print(data)
        # print(data["DATABASE"]["SERVER"])
        return data


def apply_label_encoders(data, columns):
    encoded_data = data.copy()
    for column in columns:
        label_enc_filename = os.path.join(config_dict["FILE_LOCATION"]["UTILS_FUNCTION_PATH"],f'label_encoder_{column}.pkl')
        with open(label_enc_filename, 'rb') as file:
            le = pickle.load(file)
        encoded_data[column] = le.transform(np.array(data[column]).reshape(-1,1))
    return encoded_data

def preprocessing_fxn(df):
    df.order_day = pd.to_datetime(df.order_day, format='%Y-%m-%d')
    ## Drop home_value & Pool
    ## Frency encoding sap_prdocutname
    ## city, zipcoe, county,dma -- top 3 + others

    ## Sometimes a hybrid approach can be useful. For example, you might treat year as a numeric feature to capture linear trends over time, while treating day of the week and month as categorical to capture their cyclical nature.

    ## Date Features
    df['OrderYear'] = df.order_day.dt.year
    df['OrderMonth'] = df.order_day.dt.month
    df['OrderDay'] = df.order_day.dt.day
    df['OrderDayOfYear'] = df.order_day.dt.dayofyear
    df['OrderDayOfWeek'] = df.order_day.dt.dayofweek
    df['IsWeekend'] = df['OrderDayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    ## tos_flg
    df.loc[df.tos_flg.isnull(), 'tos_flg'] = 'N'

    ## term_length
    df['term_length'] = pd.to_numeric(df['term_length'], errors='coerce').fillna(0).astype(int)

    ## curr_usage
    df['curr_usage'].fillna(0, inplace=True)

    ## risk_level (filling missing values by mode)
    df.risk_level.fillna(df.risk_level.mode().values[0], inplace=True)

    ## sap_productname (filling missing values by mode)
    df.sap_productname.fillna(df.sap_productname.mode().values[0], inplace=True)

    ## deposit_onhand_amt
    df['deposit_onhand_amt'].fillna(0, inplace=True)

    ## Convert features to correct datatype
    df['zipcode'] = df['zipcode'].apply(str)
    df['segment'] = df['segment'].apply(str)
    df['OrderMonth'] = df['OrderMonth'].apply(str)
    df['OrderDayOfWeek'] = df['OrderDayOfWeek'].apply(str)

    ## generating Features list, dropping features like 'pool', 'home_value', 'meter_id', 'customer_id', 'order_day'
    ignore_cols = ['pool', 'home_value', 'meter_id', 'customer_id', 'order_day']
    features = [i for i in df.columns if i not in ignore_cols]
    # print(features)
    txt = "# Features={} \n\n Features={}"
    logging.info(txt.format(len(features),features))

    X = df[features]

    ## sap_productname_mapping

    sap_mapping_file_name = os.path.join(config_dict["FILE_LOCATION"]["UTILS_FUNCTION_PATH"],'sap_productname_freq_map.json')
    # print(mapping_file_name)
    # print('yes')
    with open(sap_mapping_file_name, 'r') as file:
        loaded_map = json.load(file)

    # Apply the mapping
    X['sap_productname'] = X['sap_productname'].map(loaded_map).fillna(0)
    X['sap_productname'] = X['sap_productname'].astype(np.int64)

    ## Catergory Mapping

    ctgy_mapping_file_name = os.path.join(config_dict["FILE_LOCATION"]["UTILS_FUNCTION_PATH"],
                                         'top_categories.json')
    # Load the mapping in the production environment
    with open(ctgy_mapping_file_name, 'r') as file:
        top_categories = json.load(file)

    # Function to apply the mapping
    def apply_top_category_mapping(series, top_categories):
        return series.apply(lambda x: x if x in top_categories else 'Other')

    # Apply the mapping
    X['city'] = apply_top_category_mapping(X['city'], top_categories['top_3_city'])
    X['zipcode'] = apply_top_category_mapping(X['zipcode'], top_categories['top_3_zipcode'])
    X['county'] = apply_top_category_mapping(X['county'], top_categories['top_3_county'])
    X['dma'] = apply_top_category_mapping(X['dma'], top_categories['top_3_dma'])


    # print(X.dtypes)
    # Identifying numerical and categorical columns
    num_cols = X.select_dtypes(include=['int64', 'float64','int32']).columns

    ctgy_cols = X.select_dtypes(include=['object']).columns
    # print('num Columns: {} \n\nctgy columns: {}'.format(num_cols, ctgy_cols))

    txt = "Num Columns={}\n\nctgy columns: {}"
    logging.info(txt.format(num_cols.to_list(),ctgy_cols.to_list()))

    # label Encoding Categorical features

    label_encoded_features = apply_label_encoders(X, ctgy_cols)

    ## Min max Scaler

    minmax_scaler_filename= os.path.join(config_dict["FILE_LOCATION"]["UTILS_FUNCTION_PATH"],
                                         'minmax_scaler.pkl')
    with open(minmax_scaler_filename, 'rb') as file:
        loaded_scaler = pickle.load(file)

    X_num_scaled = loaded_scaler.transform(X[num_cols])

    X_num_scaled = pd.DataFrame(X_num_scaled, columns=num_cols, index=X.index)
    label_encoded_features.drop(num_cols, axis=1, inplace=True)
    X_final = X_num_scaled.join(label_encoded_features)

    txt = "Final features Shape={}"
    logging.info(txt.format(X_final.shape))

    return X_final


def gen_predictions(features):


    ##load
    xgb_model_loaded = pickle.load(open(config_dict["FILE_LOCATION"]["MODEL_FILENAME"], "rb"))
    foo = pd.DataFrame()
    foo['Predicted_Probability'] = xgb_model_loaded.predict_proba(features)[:, 1]
    txt = "Final Output Shape={}"
    logging.info(txt.format(foo.shape))
    foo.to_pickle(config_dict["FILE_LOCATION"]["OUTPUT_FILENAME"])


if __name__ == "__main__":
    # print("Start ...")
    start_time = time()
    config_dict = read_config()
    logging.basicConfig(
        format="%(asctime)s :: %(levelname)s :: %(funcName)s :: %(lineno)d \
        :: %(message)s",
        level=logging.INFO,
        filename=config_dict["FILE_LOCATION"]["LOG_FILENAME"],
        filemode="w",
    )

    logging.info("=========================================================")
    logging.info("Started ...")

    df = pd.read_csv(config_dict["FILE_LOCATION"]["TEST_DATA"])
    txt = "Test Data Shape={0}"
    logging.info(txt.format(df.shape))

    features = preprocessing_fxn(df)

    op = gen_predictions(features)
    total_time_taken = round((time() - start_time) )
    txt = "Total Time Taken={0} seconds"
    logging.info(txt.format(total_time_taken))
