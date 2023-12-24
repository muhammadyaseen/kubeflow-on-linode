"""

https://www.kubeflow.org/docs/components/pipelines/v2/components/container-components/

"""

import os
import sys
import warnings

import numpy as np
from typing import Tuple
import numpy.typing as npt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from classification_task import ClassificationTask
import helpers


RANDOM_SEED=19

# Sometimes sklearn and pandas generate warnings during CV -- we suppress them here for brevity
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

def load_data_for_full_model(selected_features, minio_params):

    # training_and_testing_data = np.load("./numpy-data-full.npz", allow_pickle=True)
    training_and_testing_data = helpers.get_train_and_test_data_as_numpy_arrays(
        selected_features,
        minio_params,
        resample_factors={
            "Plus":  3,
            "Premium": 3
        }
    )

    X_train = training_and_testing_data['X_train']
    Y_train = training_and_testing_data['Y_train']

    X_train_resampled = training_and_testing_data['X_train_resampled']
    Y_train_resampled = training_and_testing_data['Y_train_resampled']

    X_test = training_and_testing_data['X_test']
    Y_test = training_and_testing_data['Y_test']

    return X_train, Y_train, X_train_resampled, Y_train_resampled, X_test, Y_test

def get_model_class(model_name: str):

    classes = {
        'logistic_regression': LogisticRegression,
        'gbt': HistGradientBoostingClassifier,
        'decision_tree': DecisionTreeClassifier
    }

    # TODO: add try catch
    return classes[model_name]

def get_hparams_for_model(model_name: str):

    model_hparams = {
        'logistic_regression': [
            {
                "penalty": ["l1", "l2", "elasticnet", None],
                "C": [1., 1e-1, 1e-2],
                "class_weight": ["balanced"],
                "multi_class": ["ovr", "multinomial"],
                "l1_ratio": [0.25, 0.50, 0.75, 0.95]
            }
        ],
        'gbt': [
            {
                "max_depth": [2,3],
                "max_iter": [50, 100, 200],
                "class_weight": ["balanced"]
            }
        ],
        'decision_tree': [
            {
                "max_depth": [5,6,7,8],
                "min_samples_split": [10, 20, 50]
            }
        ]
    }

    # TODO: add try catch
    return model_hparams[model_name]

def train_full_model(
    model_name,
    model_class,
    X_train,
    Y_train,
    X_train_resampled,
    Y_train_resampled,
):

    hparams = get_hparams_for_model(model_name)

    # model without resampling
    full_model = ClassificationTask(
        model_name,
        model_class,
        {'random_state':RANDOM_SEED},
        hparams
    )

    full_model.train(X_train, Y_train, verbose=1)
    full_model_ranks = full_model.show_best_model()
    print(full_model_ranks)

    # model with resampling
    full_model_with_resampling = ClassificationTask(
        model_name,
        model_class,
        {'random_state':RANDOM_SEED},
        hparams
    )
    
    full_model_with_resampling.train(X_train_resampled, Y_train_resampled, verbose=1)
    full_model_with_resampling_ranks = full_model_with_resampling.show_best_model()
    print(full_model_with_resampling_ranks)

    return full_model, full_model_ranks, full_model_with_resampling, full_model_with_resampling_ranks

def test_full_model(
    trained_model: ClassificationTask, 
    X_test: npt.NDArray, 
    X_train: npt.NDArray = None
) -> Tuple[npt.NDArray, npt.NDArray]:

    Y_test_predicted, Y_train_predicted = trained_model.test(X_test, X_train)

    return Y_test_predicted, Y_train_predicted


if __name__ == "__main__":

    model_name = sys.argv[1]
    
    selected_features = [
        'search_views_per_day_in_stock',
        'detail_views_per_day_in_stock',
        'stock_days_cleaned',
        'age',
        'price_relative',
        'ctr_cleaned',
        'weekends_while_in_stock'
    ]
    
    minio_params = {
        "ip":       sys.argv[2],
        "port":     sys.argv[3],
        "bucket":   sys.argv[4],
        "object":   sys.argv[5]
    }

    # get data
    X_train, Y_train, X_train_resampled, Y_train_resampled, X_test, Y_test = load_data_for_full_model(selected_features, minio_params)

    full_model, full_model_ranks, full_model_with_resampling, full_model_with_resampling_ranks = train_full_model(
        model_name,
        get_model_class(model_name),
        X_train, 
        Y_train,
        X_train_resampled,
        Y_train_resampled
    )

    Y_test_predicted, Y_train_predicted = test_full_model(full_model, X_test, X_train)

    Y_test_predicted, Y_train_predicted = test_full_model(full_model_with_resampling, X_test, X_train)

    print(full_model)
    print(full_model_ranks)

    print(full_model_ranks)
    print(full_model_with_resampling_ranks)

    helpers.save_best_model_and_hparams_to_minio(
        full_model,
        full_model_ranks,
        {
            **minio_params,
            "bucket_name": sys.argv[4],
            "object_name_df": f"{model_name}_ranks.pandas_df",
            "object_name_model": f"{model_name}_model.obj"
        }
    )

    helpers.save_best_model_and_hparams_to_minio(
        full_model_with_resampling,
        full_model_with_resampling_ranks,
        {
            **minio_params,
            "bucket_name": sys.argv[4],
            "object_name_df": f"{model_name}_ranks_resampled.pandas_df",
            "object_name_model": f"{model_name}_model_resampled.obj"
        }
    )
