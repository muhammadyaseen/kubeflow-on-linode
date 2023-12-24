"""

https://www.kubeflow.org/docs/components/pipelines/v2/components/container-components/

"""
import os
import io
import warnings
import logging
import sys

from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import numpy.typing as npt
    
from minio import Minio
from minio.error import S3Error

# Preprocessing & metrics & models
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Resampling
from imblearn.over_sampling import SMOTE



import helpers
from classification_task import ClassificationTask


# Sometimes sklearn and pandas generate warnings during CV -- we suppress them here for brevity
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

RANDOM_SEED=19


def load_data_for_baseline_model(selected_features, minio_params):
    
    # training_and_testing_data = np.load("./numpy-data-baseline.npz", allow_pickle=True)

    training_and_testing_data = helpers.get_train_and_test_data_as_numpy_arrays(
        selected_features,
        minio_params,
        resample_factors=None
    )
    
    X_train = training_and_testing_data['X_train']
    Y_train = training_and_testing_data['Y_train']

    X_test = training_and_testing_data['X_test']
    Y_test = training_and_testing_data['Y_test']

    return X_train, Y_train, X_test, Y_test

def train_baseline_model(X_train: npt.NDArray, Y_train: npt.NDArray) -> Tuple[ClassificationTask, pd.DataFrame]:
    
    # define hyperparams
    hparams = [{"class_weight": ["balanced", None]}]

    # create training task
    baseline_model = ClassificationTask(
        "baseline_1d_logistic_clf_with_grid_search",
        LogisticRegression,
        {'random_state':RANDOM_SEED},
        hparams
    )

    # train
    baseline_model.train(X_train, Y_train, verbose=1)

    # get top models
    ranks = baseline_model.show_best_model()

    return baseline_model, ranks

def test_baseline_model(
    baseline_model: ClassificationTask, 
    X_test: npt.NDArray, 
    X_train: npt.NDArray = None
) -> Tuple[npt.NDArray, npt.NDArray]:

    Y_test_predicted, Y_train_predicted = baseline_model.test(X_test, X_train)

    return Y_test_predicted, Y_train_predicted

if __name__ == "__main__":

    model_name = sys.argv[1]

    selected_features = [
        'search_views_per_day_in_stock'
    ]
    
    minio_params = {
        "ip":       sys.argv[2],
        "port":     sys.argv[3],
        "bucket":   sys.argv[4],
        "object":   sys.argv[5]
    }
    
    # get data
    X_train, Y_train, X_test, Y_test = load_data_for_baseline_model(selected_features, minio_params)

    baseline_model , ranks = train_baseline_model(X_train, Y_train)

    Y_test_predicted, Y_train_predicted = test_baseline_model(baseline_model, X_test, X_train)

    print(ranks)

# docker run -p 9000:9000 myaseende/my-scikit \
#      python baseline_model.py none localhost 9000 car-tier-prediction-data data-with-features-cleaned.csv
    
# kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
# kubectl port-forward -n kubeflow svc/minio-service 9000:9000
# minikube start -p basic-kfp