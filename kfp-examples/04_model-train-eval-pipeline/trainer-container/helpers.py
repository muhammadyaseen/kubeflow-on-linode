"""

https://www.kubeflow.org/docs/components/pipelines/v2/components/container-components/

"""
import io
import logging
import pickle

from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import numpy.typing as npt
    
from minio import Minio
from minio.error import S3Error

# Preprocessing & metrics & models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Resampling
from imblearn.over_sampling import SMOTE


def get_dataframe_from_local():

    return pd.read_csv("./train-data/data-with-features-cleaned.csv")

def get_dataframe_from_minio(
    bucket: str, 
    object: str, 
    ip: str, 
    port="9000"
):
    try:

        logger = logging.getLogger('kfp_logger')
        logger.setLevel(logging.INFO)

        # Read-in CSV from MinIO
        # Create client with access and secret key
        client = Minio(
            f'{ip}:{port}',
            'minio',
            'minio123',
            secure=False
        )

        logger.info(f'Attempting to read data from MinIO.')

        response = client.get_object(bucket, object)

        pandas_df = pd.read_csv(io.BytesIO(response.data))

        logger.info('Columns: ' + ', '.join(pandas_df.columns))  

        return pandas_df
    
    except Exception as err:
    
        logger.error(f'Error occurred: {err}.')

def get_train_and_test_data_as_numpy_arrays(
    selected_features: List[str],
    minio_params: Dict[str, str],
    train_pct = 0.8,
    test_pct = 0.2,
    resample_factors: Dict[str, int] = None,
    random_state = 42,
):

    data_df = get_dataframe_from_minio(**minio_params)
    # data_df = get_dataframe_from_local()
    
    assert selected_features is not None and len(selected_features) > 0, "You must select at least 1 feature."

    base_X = data_df[selected_features]
    base_Y = data_df[['product_tier']]

    X_train, X_test, Y_train, Y_test = train_test_split(
        base_X, 
        base_Y,
        test_size=test_pct,
        train_size=train_pct,
        random_state=random_state
    )

    std_scalar = StandardScaler()
    X_train = std_scalar.fit_transform(X_train)
    X_test = std_scalar.transform(X_test)

    n_samples_plus_orig = np.sum(Y_train['product_tier'] == 'Plus')
    n_samples_premium_orig = np.sum(Y_train['product_tier'] == 'Premium')

    if resample_factors is not None:
        resampler = SMOTE(
            sampling_strategy={
                "Plus":  resample_factors['Plus'] * n_samples_plus_orig,
                "Premium": resample_factors['Premium'] * n_samples_premium_orig
            },
            random_state=random_state
        )

        X_train_resampled, Y_train_resampled = resampler.fit_resample(X_train, Y_train)
    
    else:
        X_train_resampled, Y_train_resampled = None, None
    
    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_train_resampled": X_train_resampled,
        "Y_train_resampled": Y_train_resampled,
        "X_test": X_test,
        "Y_test": Y_test
    }


def save_best_model_and_hparams_to_minio(
    best_model, 
    best_model_hparams_df,
    minio_params: Dict[str, str]
):
    
    try:

        logger = logging.getLogger('kfp_logger')
        logger.setLevel(logging.INFO)

        # Read-in CSV from MinIO
        # Create client with access and secret key
        client = Minio(
            f"{minio_params['ip']}:{minio_params['port']}",
            'minio',
            'minio123',
            secure=False
        )
    
        # Upload the hparams dataframe as an object.
        encoded_df = best_model_hparams_df.to_csv(index=False).encode('utf-8')
        client.put_object(
            minio_params["bucket_name"], 
            minio_params["object_name_df"], 
            data=io.BytesIO(encoded_df), 
            length=len(encoded_df), 
            content_type='application/csv'
        )
        
        logger.info(f'{minio_params["object_name_df"]} successfully uploaded to bucket {minio_params["bucket_name"]}.')
        logger.info(f'Object length: {len(best_model_hparams_df)}.')
        logger.info('Columns: ' + ', '.join(best_model_hparams_df.columns))  

        # Upload the model/class object as pkl
        object_as_bytes = pickle.dumps(best_model)
        client.put_object(
            minio_params["bucket_name"], 
            minio_params["object_name_model"], 
            data=io.BytesIO(object_as_bytes), 
            length=len(object_as_bytes) 
        )

        logger.info(f'{minio_params["object_name_model"]} successfully uploaded to bucket {minio_params["bucket_name"]}.')
        logger.info(f'Object length: {len(object_as_bytes)}.')
    
    except Exception as err:
    
        logger.error(f'Error occurred: {err}.')
