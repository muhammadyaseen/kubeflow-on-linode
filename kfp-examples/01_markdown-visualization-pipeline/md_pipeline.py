"""
A simple pipeline to show some Markdown visualizations

For some reason, Kubeflow Pipelines UI can't render Markdown tables correctly when 
python's triple quoted strings are used. This is the reason you are seeing these long strings.

For details, see this issue: https://github.com/kubeflow/pipelines/issues/10182
"""

import kfp
from kfp.dsl import Output, Dataset, Input, Markdown


@kfp.dsl.component
def write_simple_markdown_table(markdown_artifact: Output[Markdown]):

    markdown_content = "| Num   | animal1   | animal_2   | \n |---|-----------|-----------| \n |  0 | elk        | dog        | \n |  1 | pig        | quetzal    | "

    with open(markdown_artifact.path, 'w') as f:
        f.write(markdown_content)

@kfp.dsl.component
def write_simple_markdown_heading(markdown_artifact: Output[Markdown]):
    
    markdown_content = '## Hello world \n\n Markdown content'
    with open(markdown_artifact.path, 'w') as f:
        f.write(markdown_content)

@kfp.dsl.component
def vertex_ai_markdown_example(md_artifact: Output[Markdown]):

    table_md = "## Table Visualization \n | ID 	| Name  	| Type   	| Correctness 	| Count 	|  \n  |----	|:-------:	|:--------:	|:-------------:	|:-------:	|  \n | 1  	| Apple 	| Fruit  	| 60%         	| 40    	|  \n  | 2  	| Cat   	| Animal 	| 30%         	| 156   	|  \n  | 3  	| Dog   	| Animal 	| 90%         	| 592   	|"

    with open(md_artifact.path, 'w') as f:
        f.write(table_md)


@kfp.dsl.component(packages_to_install=['pandas'])
def write_pandas_dataframe_as_markdown(df_as_md: Output[Markdown]):

    import pandas as pd
    
    df = pd.DataFrame(
        data={"animal_1": ["elk", "pig"], "animal_2": ["dog", "quetzal"]}
    )
    
    df_as_md_str = df.to_markdown()
    
    with open(df_as_md.path,'w') as f:
        f.write(df_as_md_str)


@kfp.dsl.pipeline(
    name='md-pipeline',
    description='Markdown Pipeline.'
)
def markdown_pipeline():
    
    component1 = write_simple_markdown_table()
    component2 = write_simple_markdown_heading()
    component3 = vertex_ai_markdown_example()
    component3 = write_pandas_dataframe_as_markdown()


if __name__ == "__main__":

    kfp.compiler.Compiler().compile(
        pipeline_func=markdown_pipeline,
        package_path="md-pipeline.yaml"
    )
