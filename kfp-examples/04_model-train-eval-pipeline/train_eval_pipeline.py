import kfp


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

    _ = say_hello()

    train_eval_baseline_model(
        ip=ip,
        port=port,
        bucket_name=bucket_name,
        object_name=object_name
    ).after(say_hello)


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
