import kfp
from kfp.dsl import Output, Dataset, Input, Markdown

@kfp.dsl.container_component
def say_hello():
    return kfp.dsl.ContainerSpec(image='alpine', command=['echo'], args=['Hello'])

@kfp.dsl.container_component
def train_eval_baseline_model(
    ip: str,
    port: str,
    bucket_name: str,
    object_name: str
):
    
    return kfp.dsl.ContainerSpec(
        image='myaseende/my-scikit:latest', 
        command=['python'], 
        args=[
            'baseline_model.py',
            'none',
            ip,
            port,
            bucket_name,
            object_name
        ]
    )

@kfp.dsl.component(packages_to_install=['pandas', 'minio==7.1.14'])
def show_best_model_info(
    ip: str,
    port: str,
    bucket_name: str, 
    object_name: str,
    hparams_as_md: Output[Markdown]
) -> float:

    import logging
    import io

    from minio import Minio
    from minio.error import S3Error
    import pandas as pd

    try:

        logger = logging.getLogger('kfp_logger')
        logger.setLevel(logging.INFO)

        # Create client with access and secret key
        client = Minio(
            f'{ip}:{port}',
            'minio',
            'minio123',
            secure=False
        )

        logger.info(f'Attempting to read data from MinIO.')

        # read the ranks and hparams info df from MinIO
        response = client.get_object(bucket_name, object_name)

        pandas_df = pd.read_csv(io.BytesIO(response.data))

        logger.info('Columns: ' + ', '.join(pandas_df.columns))  

        with open(hparams_as_md.path, 'w') as f:
            f.write(pandas_df.to_markdown())
        
        metric = pandas_df.loc[0, 'mean_test_balanced_accuracy'].item()
        logger.info(f"Metric (mean_test_balanced_accuracy) on Baseline: {metric}")

        return metric
    
    except Exception as err:
    
        logger.error(f'Error occurred: {err}.')
    
    # output the main metric (this can be used for comparison)
    return 0.0

@kfp.dsl.pipeline(
   name='base-train-eval-pipeline',
   description='Pipeline that will train a baseline model and run eval'
)
def model_train_eval_pipeline(
    ip: str,
    port: str,
    bucket_name: str,
    object_name: str
):

    greeting_task = say_hello()

    # This will train, eval, and save the model as well as hparams
    base_line_train_eval_task = train_eval_baseline_model(
        ip=ip,
        port=port,
        bucket_name=bucket_name,
        object_name=object_name
    ).after(greeting_task)

    base_line_train_eval_task.set_caching_options(False)

    metric = show_best_model_info(
        ip=ip,
        port=port,
        bucket_name=bucket_name,
        object_name="baseline_model_ranks.pandas_df"
    ).after(base_line_train_eval_task)

    metric.set_caching_options(False)

if __name__ == "__main__":

    kfp.compiler.Compiler().compile(
        pipeline_func=model_train_eval_pipeline,
        package_path="model_train_eval_pipeline.yaml",
        pipeline_parameters={
            'bucket_name': 'car-tier-prediction-data',
            'object_name': 'data-with-features-cleaned.csv',
            'port': '9000'
        }
    )
