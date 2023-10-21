import os

from typing import List, Tuple, Union, Dict

# External libs
import pandas as pd
import numpy as np

# Visualizations
from sklearn.utils import class_weight
import seaborn as sns
import matplotlib.pyplot as plt


# Models
from sklearn.linear_model import ( ElasticNet, LogisticRegression,
                                  Ridge, Lasso, LinearRegression)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.svm import SVC, SVR

# Preprocessing & metrics
from sklearn.preprocessing import (OneHotEncoder, StandardScaler)
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
  r2_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve,
  roc_curve, balanced_accuracy_score
)
from sklearn.model_selection import (train_test_split, cross_validate,
                                     GridSearchCV)
from sklearn.compose import make_column_selector, ColumnTransformer

# Resampling
from imblearn.over_sampling import (SMOTE, SMOTENC, RandomOverSampler)

# Sometimes sklearn and pandas generate warnings during CV -- we suppress them here for brevity
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

def read_csv_from_minio(
    minio_endpoint: str,
    minio_secret: str,
    minio_key: str,
    bucket_name: str,
    object_name: str
):
    """
    
    """

    import minio
    import pandas as pd

    from io import StringIO
    
    minio_client = minio.Minio(
        minio_endpoint, 
        minio_secret,
        minio_key,
        secure=False
    )

    # This returns binary encoded data
    response = minio_client.get_object(bucket_name, object_name)
    decoded_csv_string = response.data.decode('utf-8')
    
    return pd.read_csv(StringIO(decoded_csv_string))

def prepare_cleaned_dataset(df):

    cleaned_df = df.copy(deep=True)

    # These limits were taken from the quantile dataframes. Ideally we should pass-in the dataframe
    # and extract the limits automatically.
    cleaned_df = df[
    ((df['price'] >= 1100) & (df['price'] <= 51950)) & \
    ((df['ctr_cleaned'] >= 0.006450) & (df['ctr_cleaned'] <= 0.130435)) & \
    ((df['search_views'] >= 63) & (df['search_views'] <= 11119.00)) & \
    ((df['detail_views'] >= 1) & (df['detail_views'] <= 900)) & \
    ((df['stock_days'] >= 1) & (df['stock_days'] <= 112)) & \
    ((df['first_registration_year'] >= 1989) & (df['first_registration_year'] <= 2023)) &
    (df['detail_views_per_day_in_stock'] <= 25)  | \
    ((df['product_tier'] == 'Premium') | (df['product_tier'] == 'Plus')) # we don't want to loose precious examples of minority classes
    ]

    # TODO: write this df as CSV in MinIO

    return cleaned_df

def make_dervied_features(df):

    import pandas as pd

    def _get_days(date_str_future, date_str_past):
        
        from datetime import datetime
        
        f = datetime.strptime(date_str_future, "%d.%m.%y").date()
        p = datetime.strptime(date_str_past, "%d.%m.%y").date()
        
        return (f - p).days + 1

    def _get_ctr(search_views, detail_views):

        if search_views == 0:
            return 0.
        else:
            return detail_views / search_views


    # TODO: Read from MinIO
    # TODO: Create Pandas df
    

    df['ctr_cleaned'] = df.apply(
        lambda x: _get_ctr(
            x['search_views'],
            x['detail_views']
        ), axis=1
    )
    df['stock_days_cleaned'] = df.apply(
        lambda x: _get_days(
            x['deleted_date'],
            x['created_date']
        ), axis=1
    )

    # If an entry has been up throughout the weekend, it may have been viewed / searched for more times.
    df['weekends_while_in_stock'] = df['stock_days_cleaned'] // 7

    df['age'] = df['first_registration_year'].max() - df['first_registration_year']
    df['price_k'] = df['price'] / 1000

    # Cars are not compared in isolation, visitors compare them with what's available
    df['price_relative'] = df['price_k'] / df['price_k'].mean()

    # Normalizing by in stock days
    df['search_views_per_day_in_stock'] = df['search_views'] / df['stock_days_cleaned']
    df['detail_views_per_day_in_stock'] = df['detail_views'] / df['stock_days_cleaned']

    # not needed
    df['search_and_detail_impressions_diff'] = df['search_views_per_day_in_stock'] - df['detail_views_per_day_in_stock']

    # TODO: Write back to MinIO

    return df


def product_tier_counts_and_percentages(df):

    # TODO: read from minio

    # TODO: show on KFP UI

    product_tier_counts_df = df[['product_tier', 'article_id']].groupby(['product_tier']).count()

    product_tier_percentages_df = product_tier_counts_df * 100 / len(df)

    return product_tier_counts_df, product_tier_percentages_df


def get_training_and_test_data_for_classification_task(
    X_raw,
    Y_raw,
    *,
    selected_features: List[str] = None,
    train_pct = 0.8,
    test_pct = 0.2,
    resample = False,
    resample_factors: Dict[str, int] = None,
    random_state = 42
):

    assert selected_features is not None and len(selected_features) > 0, "You must select at least 1 feature."

    RANDOM_SEED = 1234
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_raw[selected_features], Y_raw,
        test_size=test_pct,
        train_size=train_pct,
        random_state=random_state
    )

    std_scalar = StandardScaler()
    X_train = std_scalar.fit_transform(X_train)
    X_test = std_scalar.transform(X_test)

    if resample:

        n_samples_plus_orig = np.sum(Y_train['product_tier'] == 'Plus')
        n_samples_premium_orig = np.sum(Y_train['product_tier'] == 'Premium')

        resampler = SMOTE(
            sampling_strategy={
                "Plus":  resample_factors['Plus'] * n_samples_plus_orig,
                "Premium": resample_factors['Premium'] * n_samples_premium_orig
            },
            random_state=RANDOM_SEED
        )

        X_train, Y_train = resampler.fit_resample(X_train, Y_train)

    return X_train, Y_train, X_test, Y_test

