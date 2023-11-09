"""
Adapted from: https://blog.min.io/building-an-ml-data-pipeline-with-minio-and-kubeflow-v2-0/
"""

import kfp
import minio

from kfp.dsl import Output, Dataset, Input


@kfp.dsl.component(packages_to_install=['minio==7.1.14'])
def check_if_table_data_exists_already(bucket: str, table_code: str, year: int) -> bool:
    '''
    Check for the existence of Census table data in the given MinIO bucket.
    '''
    from minio import Minio
    from minio.error import S3Error
    import logging

    object_name=f'{table_code}-{year}.csv'

    logger = logging.getLogger('kfp_logger')
    logger.setLevel(logging.INFO)
    logger.info(bucket)
    logger.info(table_code)
    logger.info(year)
    logger.info(object_name)
  
    try:
        # Create client with access and secret key.
        client = Minio('minio-service-ip-here:9000',
                    'minio',
                    'minio123',
                    secure=False)

        # First, check if the bucket exists
        bucket_found = client.bucket_exists(bucket)
        if not bucket_found:
            return False

        # If so, check if the file / object exists that contains the data
        objects = client.list_objects(bucket)
        found = False
        
        for obj in objects:
            
            logger.info(obj.object_name)
            
            if object_name == obj.object_name: 
                found = True

    except S3Error as s3_err:
        logger.error(f'S3 Error occurred: {s3_err}.')
    except Exception as err:
        logger.error(f'Error occurred: {err}.')

    return found


@kfp.dsl.component(packages_to_install=['pandas==1.3.5', 'requests'])
def download_table_data(dataset: str, table_code: str, year: int, table_df: Output[Dataset]):
    '''
    Returns all fields for the specified table. The output is a DataFrame saved to csv.
    '''
    import logging
    import pandas as pd
    import requests

    logger = logging.getLogger('kfp_logger')
    logger.setLevel(logging.INFO)

    # e.g. dataset = acs
    # https://www.census.gov/data/developers/data-sets/acs-5year/2021.html
    census_endpoint = f'https://api.census.gov/data/{year}/{dataset}/acs5'
    census_key = 'us-census-key-here'

    # Setup a simple dictionary for the requests parameters.
    get_token = f'group({table_code})'
    params = {'key': census_key, 'get': get_token, 'for': 'county:*'}

    # sending get request and saving the response as response object
    response = requests.get(url=census_endpoint, params=params)

    # Extract the data in json format.
    # The first row of our matrix contains the column names. The remaining rows
    # are the data.
    survey_data = response.json()
    
    # Create a pandas df with the response data and write it as CSV
    df = pd.DataFrame(survey_data[1:], columns = survey_data[0])
    df.to_csv(table_df.path, index=False)

    logger.info(f'Table {table_code} for {year} has been downloaded.')


@kfp.dsl.component(packages_to_install=['pandas==1.3.5', 'minio==7.1.14'])
def save_table_data(bucket: str, table_code: str, year: int, table_df: Input[Dataset]):
    '''
    Save the data in the diven MinIO Bucket.
    The input param `table_df` refers to the Dataset written by `download_table_data` func to an 
    intermediate location in pipeline root
    '''
    
    import io
    import logging
    from minio import Minio
    from minio.error import S3Error
    import pandas as pd

    object_name=f'{table_code}-{year}.csv'

    logger = logging.getLogger('kfp_logger')
    logger.setLevel(logging.INFO)
    logger.info(bucket)
    logger.info(table_code)
    logger.info(year)
    logger.info(object_name)

    # Read in the data that was written by `download_table_data` func
    df = pd.read_csv(table_df.path)

    try:
        # Create client with access and secret key
        client = Minio('minio-service-ip-here:9000',
                    'minio',
                    'minio123',
                    secure=False)

        # Make the bucket if it does not exist.
        found = client.bucket_exists(bucket)
        if not found:
            logger.info(f'Creating bucket: {bucket}.')
            client.make_bucket(bucket)

        # Upload the dataframe as an object.
        encoded_df = df.to_csv(index=False).encode('utf-8')
        client.put_object(bucket, object_name, data=io.BytesIO(encoded_df), length=len(encoded_df), content_type='application/csv')
        logger.info(f'{object_name} successfully uploaded to bucket {bucket}.')
        logger.info(f'Object length: {len(df)}.')

    except S3Error as s3_err:
        logger.error(f'S3 Error occurred: {s3_err}.')
    except Exception as err:
        logger.error(f'Error occurred: {err}.')


@kfp.dsl.component(packages_to_install=['pandas==1.3.5', 'minio==7.1.14'])
def get_table_data(bucket: str, table_code: str, year: int, table_df: Output[Dataset]):
    '''
    Read CSV data from MinIO into a Pandas df
    '''
    
    import io
    import logging
    from minio import Minio
    from minio.error import S3Error
    import pandas as pd

    object_name=f'{table_code}-{year}.csv'

    logger = logging.getLogger('kfp_logger')
    logger.setLevel(logging.INFO)
    logger.info(bucket)
    logger.info(table_code)
    logger.info(year)
    logger.info(object_name)

    # Get data of an object.
    try:
        # Create client with access and secret key
        client = Minio('minio-service-ip-here:9000',
                    'minio',
                    'minio123',
                    secure=False)

        response = client.get_object(bucket, object_name)
        df = pd.read_csv(io.BytesIO(response.data))
        df.to_csv(table_df.path, index=False)
        logger.info(f'Object: {object_name} has been retrieved from bucket: {bucket} in MinIO object storage.')
        logger.info(f'Object length: {len(df)}.')

    except S3Error as s3_err:
        logger.error(f'S3 Error occurred: {s3_err}.')
    except Exception as err:
        logger.error(f'Error occurred: {err}.')

    finally:
        response.close()
        response.release_conn()


@kfp.dsl.pipeline(
   name='census-pipeline',
   description='Pipeline that will download Census data and save to MinIO.'
)
def census_pipeline(bucket: str, dataset: str, table_code: str, year: int) -> Dataset:
    # Positional arguments are not allowed.
    # When I set the name parameter of the condition that task in the DAG fails.

    exists = check_if_table_data_exists_already(bucket=bucket, table_code=table_code, year=year)

    with kfp.dsl.Condition(exists.output == False):
        
        table_data = download_table_data(dataset=dataset, table_code=table_code, year=year)
        
        save_table_data(
            bucket=bucket,
            table_code=table_code,
            year=year,
            table_df=table_data.outputs['table_df']
        )

    with kfp.dsl.Condition(exists.output == True):
        table_data = get_table_data(
            bucket=bucket,
            table_code=table_code,
            year=year
        )

    return table_data.outputs['table_df']


if __name__ == "__main__":

    kfp.compiler.Compiler().compile(
        pipeline_func=census_pipeline,
        package_path="minio-census-pipeline.yaml",
        pipeline_parameters={
            'bucket': 'census-data',
            'table_code': 'B01001',
            'year': 2021
        }
    )
