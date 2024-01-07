import json
from typing import Dict
import kfp
from kfp.dsl import Output, Dataset, Input, Markdown


@kfp.dsl.container_component
def train_eval_baseline_model(
    model_name: str,
    script: str,
    ip: str,
    port: str,
    bucket_name: str,
    object_name: str
):
    """
    This component loads our main training container image and runs the code to train the models
    """

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

@kfp.dsl.component(packages_to_install=['pandas'])
def find_best_model_on_full_data(
    baseline_metric: float,
    lr_metric: float,
    lr_resampled_metric: float,
    gbt_metric: float,
    gbt_resampled_metric: float,
    dtree_metric: float,
    dtree_resampled_metric: float,
    experiment_summary: Output[Markdown]
):
    """
    Given the evaluation scores, find out which model did the best
    """

    import logging
    import pandas as pd

    logger = logging.getLogger('kfp_logger')
    logger.setLevel(logging.INFO)
    
    # TODO: there should be a better way to do this
    model_names = [
        'baseline',
        'lr',
        'lr_resampled',
        'gbt',
        'gbt_resampled',
        'dtree',
        'dtree_resampled'
    ]

    model_metrics = [
        baseline_metric,
        lr_metric,
        lr_resampled_metric,
        gbt_metric,
        gbt_resampled_metric,
        dtree_metric,
        dtree_resampled_metric
    ]
    
    models_and_metrics = dict(
        zip(
            model_names,
            model_metrics
        )
    )

    best_model_name = max(models_and_metrics, key=lambda key: models_and_metrics[key])
    best_model_metric = models_and_metrics[best_model_name]

    # write experiments summary as a markdown table

    logger.info('Writing experiment summary')
    
    with open(experiment_summary.path, 'w') as f:
        
        experiment_summary_df = pd.DataFrame({
            # We replace the underscore so that Mardown is rendered properly
            "Models": list(map(lambda name: name.replace("_", " "), model_names)),
            "Metrics": model_metrics,
        })

        summaries_table_md = experiment_summary_df.to_markdown()
        summaries_table_md += "\n\n"
        summaries_table_md += f"The best model is **{best_model_name}** with evaluation metric value: **{best_model_metric}.**"
        
        f.write(summaries_table_md)
    

@kfp.dsl.component(packages_to_install=['pandas', 'minio==7.1.14'])
def show_best_model_info(
    model_name: str,
    ip: str,
    port: str,
    bucket_name: str, 
    object_name: str,
) -> float:
    """
    Retrieve the saved training metrics
    """

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
):

    # Note: the `.set_caching_options(False)` call is used so that we don't reuse results from previous runs.
    # This was required during developmenet because sometimes I change the container and push it to DockerHub, but 
    # Kubeflow wasn't pulling the new container and was instead using old results.

    ############################
    # Baseline model
    ############################

    # This will train, eval, and save the model as well as hparams
    base_line_train_eval_task = train_eval_baseline_model(
        model_name='baseline',
        script='baseline_model.py',
        ip=ip,
        port=port,
        bucket_name=bucket_name,
        object_name=object_name
    )

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
    
    # 1. Logistic regression + with resampling
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

    # 2. Gradient Boosted Trees regression + with resampling
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

    # Decision Tree regression + with resampling
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
    # TODO: For some reason, collecting the results into a dictionary like this doesn't seem to work

    # model_and_metric_dict={
    #     "baseline":         base_line_metric.output,
    #     "lr":               lr_metric.output,
    #     "lr_resampled":     lr_resampled_metric.output,
    #     "gbt":              gbt_metric.output,
    #     "gbt_resampled":    gbt_resampeld_metric.output,
    #     "dtree":            dtree_metric.output,
    #     "dtree_resampled":  dtree_resampled_metric.output
    # }
    
    # TODO: There has to be a better way of doing this, but for now we go with this (honestly) ugly solution
    find_best_model_on_full_data(
        baseline_metric=base_line_metric.output,
        lr_metric=lr_metric.output,
        lr_resampled_metric=lr_resampled_metric.output,
        gbt_metric=gbt_metric.output,
        gbt_resampled_metric=gbt_resampeld_metric.output,
        dtree_metric=dtree_metric.output,
        dtree_resampled_metric=dtree_resampled_metric.output
    )


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
