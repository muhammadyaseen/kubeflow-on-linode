import json
from typing import Dict
import kfp
from kfp.dsl import Output, Dataset, Input, Markdown

@kfp.dsl.container_component
def say_hello():
    return kfp.dsl.ContainerSpec(image='alpine', command=['echo'], args=['Hello'])

@kfp.dsl.container_component
def train_eval_baseline_model(
    model_name: str,
    script: str,
    ip: str,
    port: str,
    bucket_name: str,
    object_name: str
):

    return kfp.dsl.ContainerSpec(
        image='myaseende/my-scikit:latest', 
        command=['python'], 
        args=[
            script,
            model_name,
            ip,
            port,
            bucket_name,
            object_name
        ]
    )

@kfp.dsl.component
def find_best_model_on_full_data(
    baseline_metric: float,
    lr_metric: float
) -> str:
    
    # models_and_metrics_dict = json.loads(models_and_metrics)

    #  "baseline":         base_line_metric.output,
    # "lr":               lr_metric.output,
    # "lr_resampled":     lr_resampled_metric.output,
    # "gbt":              gbt_metric.output,
    # "gbt_resampled":    gbt_resampeld_metric.output,
    # "dtree":            dtree_metric.output,
    # "dtree_resampled":  dtree_resampled_metric.output

    models_and_metrics = dict(
        zip(
            [
                'baseline',
                'lr'
            ],
            [
                baseline_metric,
                lr_metric
            ]
        )
    )
    
    return max(models_and_metrics, key=lambda key: models_and_metrics[key])


@kfp.dsl.component(packages_to_install=['pandas', 'minio==7.1.14'])
def show_best_model_info(
    model_name: str,
    ip: str,
    port: str,
    bucket_name: str, 
    object_name: str,
#    hparams_as_md: Output[Markdown]
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

        #with open(hparams_as_md.path, 'w') as f:
        #    f.write(pandas_df.to_markdown())
        
        metric = pandas_df.loc[0, 'mean_test_balanced_accuracy'].item()
        logger.info(f"Metric (mean_test_balanced_accuracy) on {model_name}: {metric}")

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
) -> str:


    ############################
    # Baseline model
    ############################
    greeting_task = say_hello()

    # This will train, eval, and save the model as well as hparams
    base_line_train_eval_task = train_eval_baseline_model(
        model_name='baseline',
        script='baseline_model.py',
        ip=ip,
        port=port,
        bucket_name=bucket_name,
        object_name=object_name
    ).after(greeting_task)

    base_line_train_eval_task.set_caching_options(False)

    base_line_metric = show_best_model_info(
        model_name='baseline',
        ip=ip,
        port=port,
        bucket_name=bucket_name,
        object_name="baseline_model_ranks.pandas_df"
    ).after(base_line_train_eval_task)

    base_line_metric.set_caching_options(False)


    ############################
    # Full models
    ############################
    
    # logistic reg
    lr_train_eval_task = train_eval_baseline_model(
        model_name='logistic_regression',
        script='full_models.py',
        ip=ip,
        port=port,
        bucket_name=bucket_name,
        object_name=object_name
    )
    lr_train_eval_task.set_caching_options(False)

    lr_metric = show_best_model_info(
        model_name='logistic_regression',
        ip=ip,
        port=port,
        bucket_name=bucket_name,
        object_name="logistic_regression_ranks.pandas_df"
    ).after(lr_train_eval_task)

    lr_resampled_metric = show_best_model_info(
        model_name='logistic_regression_resampled',
        ip=ip,
        port=port,
        bucket_name=bucket_name,
        object_name="logistic_regression_ranks_resampled.pandas_df"
    ).after(lr_train_eval_task)

    lr_metric.set_caching_options(False)
    lr_resampled_metric.set_caching_options(False)

    # # gbt reg
    gbt_train_eval_task = train_eval_baseline_model(
        model_name='gbt',
        script='full_models.py',
        ip=ip,
        port=port,
        bucket_name=bucket_name,
        object_name=object_name
    )
    gbt_train_eval_task.set_caching_options(False)

    gbt_metric = show_best_model_info(
        model_name='gbt',
        ip=ip,
        port=port,
        bucket_name=bucket_name,
        object_name="gbt_ranks.pandas_df"
    ).after(gbt_train_eval_task)
    
    gbt_resampeld_metric = show_best_model_info(
        model_name='gbt_resampled',
        ip=ip,
        port=port,
        bucket_name=bucket_name,
        object_name="gbt_ranks_resampled.pandas_df"
    ).after(gbt_train_eval_task)

    gbt_metric.set_caching_options(False)
    gbt_resampeld_metric.set_caching_options(False)

    # # decision_tree reg
    dtree_train_eval_task = train_eval_baseline_model(
        model_name='decision_tree',
        script='full_models.py',
        ip=ip,
        port=port,
        bucket_name=bucket_name,
        object_name=object_name
    )
    dtree_train_eval_task.set_caching_options(False)

    dtree_metric = show_best_model_info(
        model_name='decision_tree',
        ip=ip,
        port=port,
        bucket_name=bucket_name,
        object_name="decision_tree_ranks.pandas_df"
    ).after(dtree_train_eval_task)

    dtree_resampled_metric = show_best_model_info(
        model_name='decision_tree_resampled',
        ip=ip,
        port=port,
        bucket_name=bucket_name,
        object_name="decision_tree_ranks_resampled.pandas_df"
    ).after(dtree_train_eval_task)

    dtree_metric.set_caching_options(False)
    dtree_resampled_metric.set_caching_options(False)

    # ############################
    # # Compare models
    # ############################
    # model_and_metric_dict={
    #     "baseline":         base_line_metric.output,
    #     "lr":               lr_metric.output,
    #     "lr_resampled":     lr_resampled_metric.output,
    #     "gbt":              gbt_metric.output,
    #     "gbt_resampled":    gbt_resampeld_metric.output,
    #     "dtree":            dtree_metric.output,
    #     "dtree_resampled":  dtree_resampled_metric.output
    # }
    best_model = find_best_model_on_full_data(
        baseline_metric=base_line_metric.output,
        lr_metric=lr_metric.output,
    )

    return best_model.output

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
