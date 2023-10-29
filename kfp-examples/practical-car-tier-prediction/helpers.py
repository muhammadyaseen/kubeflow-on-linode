import kfp
from kfp.dsl import Output, Dataset, Input, Markdown

@kfp.dsl.component(packages_to_install=['minio==7.1.14'])
def check_if_raw_data_exists_already(
    bucket_name: str,
    object_name: str
) -> bool:
    
    from minio import Minio
    from minio.error import S3Error
    import logging


    logger = logging.getLogger('kfp_logger')
    logger.setLevel(logging.INFO)
    logger.info(f"Looking for {object_name} in Bucket: {bucket_name}")
      
    try:
    
        # Create client with access and secret key.
        client = Minio(
            '10.110.111.225:9000',
            'minio',
            'minio123',
            secure=False
        )

        # First, check if the bucket exists
        bucket_found = client.bucket_exists(bucket_name)
        if not bucket_found:
            return False

        # If so, check if the file / object exists that contains the data
        available_objects = client.list_objects(bucket_name)
        found = False
        
        for current_object in available_objects:
            
            logger.info(current_object.object_name)
            
            if object_name == current_object.object_name: 
                found = True

    except S3Error as s3_err:
        logger.error(f'S3 Error occurred: {s3_err}.')
    except Exception as err:
        logger.error(f'Error occurred: {err}.')

    return found

@kfp.dsl.component
def download_raw_data_for_pipeline(raw_data_location: Output[Dataset]):
    
    import logging
    import urllib.request

    logger = logging.getLogger('kfp_logger')
    logger.setLevel(logging.INFO)

    CSV_PATH = "https://raw.githubusercontent.com/muhammadyaseen/kubeflow-on-linode/main/kfp-examples/practical-car-tier-prediction/data/kfp-practical-product-tier-prediction-data.csv"
    
    with urllib.request.urlopen(CSV_PATH) as raw_data_csv:
        raw_data_content = raw_data_csv.read().decode('utf-8')
       
        with open(raw_data_location.path, 'w') as f:       
            f.write(raw_data_content)

    logger.info(f'Raw data downloaded and saved at: {raw_data_location.path}')

@kfp.dsl.component(packages_to_install=['pandas', 'minio==7.1.14'])
def save_raw_data_to_bucket(
    bucket_name: str, 
    object_name: str, 
    raw_data: Input[Dataset]
):
    
    import logging
    import io

    from minio import Minio
    from minio.error import S3Error
    import pandas as pd


    logger = logging.getLogger('kfp_logger')
    logger.setLevel(logging.INFO)

    df = pd.read_csv(raw_data.path)

    try:
        # Create client with access and secret key
        client = Minio(
            '10.110.111.225:9000',
            'minio',
            'minio123',
            secure=False
        )

        # Make the bucket if it does not exist.
        if not client.bucket_exists(bucket_name):
            logger.info(f'Creating bucket: {bucket_name}.')
            client.make_bucket(bucket_name)

        # Upload the dataframe as an object.
        encoded_df = df.to_csv(index=False).encode('utf-8')
        client.put_object(
            bucket_name, 
            object_name, 
            data=io.BytesIO(encoded_df), 
            length=len(encoded_df), 
            content_type='application/csv'
        )
        
        logger.info(f'{object_name} successfully uploaded to bucket {bucket_name}.')
        logger.info(f'Object length: {len(df)}.')

    except S3Error as s3_err:
        logger.error(f'S3 Error occurred: {s3_err}.')
    except Exception as err:
        logger.error(f'Error occurred: {err}.')

@kfp.dsl.component(packages_to_install=['pandas', 'minio==7.1.14'])
def prepare_cleaned_dataset(
    bucket: str,
    object: str,
    cleaned_data: Output[Dataset]
):
    """
    
    """

    import io
    import logging
    import pandas as pd
    from minio import Minio
    from minio.error import S3Error

    try:

        logger = logging.getLogger('kfp_logger')
        logger.setLevel(logging.INFO)

        # Create client with access and secret key
        client = Minio('10.110.111.225:9000',
                    'minio',
                    'minio123',
                    secure=False)

        response = client.get_object(bucket, object)
        df = pd.read_csv(io.BytesIO(response.data))

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

        # Write this df as CSV in MinIO
        cleaned_df.to_csv(cleaned_data.path)
        
        logger.info(f'Cleaned data written to: {cleaned_data.path}')
    
    except S3Error as s3_err:
        logger.error(f'S3 Error occurred: {s3_err}.')
    except Exception as err:
        logger.error(f'Error occurred: {err}.')

    finally:
        response.close()
        response.release_conn()


@kfp.dsl.component(packages_to_install=['pandas'])
def product_tier_counts_and_percentages(
    cleaned_dataset: Input[Dataset], 
    product_percentage_md: Output[Markdown], 
    product_counts_md: Output[Markdown]
):

    import pandas as pd

    
    df = pd.read_csv(cleaned_dataset.path)

    product_tier_counts_df = df[['product_tier', 'article_id']].groupby(['product_tier']).count()

    product_tier_percentages_df = product_tier_counts_df * 100 / len(df)

    with open(product_counts_md.path, 'w') as f:
        f.write(product_tier_counts_df.to_markdown())

    with open(product_percentage_md.path, 'w') as f:
        f.write(product_tier_percentages_df.to_markdown())




@kfp.dsl.pipeline(
   name='data-preparation-cleaning-pipeline',
   description='Pipeline that will download & clean raw data and save it to MinIO.'
)
def data_preparation_pipeline(bucket: str, object: str) -> Dataset:

    exists = check_if_raw_data_exists_already(bucket_name=bucket, object_name=object)

    with kfp.dsl.Condition(
        condition=exists.output == False,
        name='Data-Does-Not-Exist'
    ):
        
        raw_data = download_raw_data_for_pipeline()
        
        save_data_to_bucket = save_raw_data_to_bucket(
            bucket_name=bucket,
            object_name=object,
            raw_data=raw_data.outputs['raw_data_location']
        )

        # Should happen after `save_raw_data_to_bucket`
        cleaned_data = prepare_cleaned_dataset(
            bucket=bucket,
            object=object
        ).after(save_data_to_bucket)

        counts_and_percent_visual = product_tier_counts_and_percentages(
            cleaned_dataset=cleaned_data.outputs['cleaned_data']
        )

    with kfp.dsl.Condition(
        condition=exists.output == True,
        name='Data-Already-Exists'
    ):

        cleaned_data = prepare_cleaned_dataset(
            bucket=bucket,
            object=object
        )
    
        counts_and_percent_visual = product_tier_counts_and_percentages(
            cleaned_dataset=cleaned_data.outputs['cleaned_data']
        )

    return cleaned_data.outputs['cleaned_data']


if __name__ == "__main__":

    kfp.compiler.Compiler().compile(
        pipeline_func=data_preparation_pipeline,
        package_path="as24-car-pipeline.yaml",
        pipeline_parameters={
            'bucket': 'as24-data',
            'object': 'as24-study.csv'
        }
    )
