# Kubeflow Pipelines on Linode

We will use the following technologies:

- Linode (Cloud / Infrastructure provider)
- Kubernetes
- Kubeflow Pipelines
- MinIO (Cloud storage)


![Banner image with logos of Linode, Kubernetes, Kubeflow, MinIO](images/banner-2.png)

<hr />

You will learn how to set up Kubeflow Pipelines on a Kubernetes cluster hosted on Linode. After this set up, you'll also get to run a simple but practical example of a pipeline which queries the US government's Census API to download some data and store it in a MinIO Bucket.

- The instructions / tutorial is available in `kubeflow-pipelines-on-linode.md` file
- Pipeline code is available in `kfp-examples` in respective folders:
    - `01_markdown-visualization-pipeline`
    - `02_simple-minio-pipeline`
    - `03_data-cleaning-pipeline`
    - `04_model-train-eval-pipeline`

- For details see: [https://muhammadyaseen.github.io/kubeflow-on-linode/](https://muhammadyaseen.github.io/kubeflow-on-linode/)

## Acknowledgements

- [MinIO Blog - Building an ML Data Pipeline with MinIO and Kubeflow v2.0](https://blog.min.io/building-an-ml-data-pipeline-with-minio-and-kubeflow-v2-0/)
- [Florian Pach's Kubeflow MNIST Pipeline on GitHub](https://github.com/flopach/digits-recognizer-kubeflow)
- [NetworkChuck's Kubernetes Video on Youtube](https://www.youtube.com/watch?v=7bA0gTroJjw)




