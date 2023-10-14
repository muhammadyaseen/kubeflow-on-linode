# Deploying Kubeflow Pipelines on Linode

## 1. Create an account on Linode and claim $100 credit

## 2. Create Kubernetes cluster

## 3. Connect to your Kubernetes cluster

- Download `kubeconfig.yaml` file from Linode
- Set up `kubectl` to use your file
```bash

export KUBECONFIG=/path/to/linode-k8s-cluster-kubeconfig.yaml

kubectl get pods -A

```

You will get output similar to the following. This shows all the pods deployed by Linode's Kubernetes installation.

```bash

NAMESPACE     NAME                                      READY   STATUS    RESTARTS   AGE
kube-system   calico-kube-controllers-d6b8dd95c-l7t4g   1/1     Running   0          179m
kube-system   calico-node-d5zx2                         1/1     Running   0          178m
kube-system   calico-node-gjjlp                         1/1     Running   0          177m
kube-system   calico-node-xrhmw                         1/1     Running   0          178m
kube-system   coredns-6989f6c969-jnnkj                  1/1     Running   0          179m
kube-system   coredns-6989f6c969-zlwpn                  1/1     Running   0          179m
kube-system   csi-linode-controller-0                   4/4     Running   0          179m
kube-system   csi-linode-node-4dxbs                     2/2     Running   0          178m
kube-system   csi-linode-node-89zfk                     2/2     Running   0          178m
kube-system   csi-linode-node-dvpbv                     2/2     Running   0          177m
kube-system   kube-proxy-995b8                          1/1     Running   0          177m
kube-system   kube-proxy-lmvxv                          1/1     Running   0          178m
kube-system   kube-proxy-wwnl6                          1/1     Running   0          178m
```

## 4. Install Kubeflow Pipelines


Now that we know that our Kubernetes cluster is up and running, we want to install Kubeflow Pipelines on it. To do that, run the following commands one by one.

```bash
export PIPELINE_VERSION=2.0.1

kubectl -- apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"

kubectl -- wait --for condition=established --timeout=60s crd/applications.app.k8s.io

kubectl -- apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"
```

These commands install the necessary Deployments, Pods, and Services as well as other Kubernetes objects that constitute Kubeflow. Note that all the objects are installed in the `kubeflow` namespace.

Since this process involves downloading and starting up a lot of container images, this can take a while. In my experience, it can take anywhere from 10 to 15 minutes before you Kubeflow installation is up and running. 

```bash
$ kubectl get pods -n kubeflow

kubeflow      cache-deployer-deployment-6f9c8677b-m8r2w          0/1     ContainerCreating   0          32s
kubeflow      cache-server-d86b59fb4-zmnmm                       0/1     ContainerCreating   0          32s
kubeflow      metadata-envoy-deployment-fd499f77f-4wwnd          1/1     Running             0          32s
kubeflow      metadata-grpc-deployment-5644fb9768-sksvz          0/1     ContainerCreating   0          31s
kubeflow      metadata-writer-7b54467cd4-47bmm                   1/1     Running             0          30s
kubeflow      minio-55464b6ddb-gsnvf                             0/1     ContainerCreating   0          30s
kubeflow      ml-pipeline-6fc946c6d9-8qgfk                       0/1     ContainerCreating   0          30s
kubeflow      ml-pipeline-persistenceagent-5958478546-lf6zn      0/1     ContainerCreating   0          30s
kubeflow      ml-pipeline-scheduledworkflow-5c4dfc67c-d8f2b      0/1     ContainerCreating   0          30s
kubeflow      ml-pipeline-ui-6df94859ff-m7m97                    0/1     ContainerCreating   0          30s
kubeflow      ml-pipeline-viewer-crd-7fccb85dd6-7xnpp            0/1     ContainerCreating   0          29s
kubeflow      ml-pipeline-visualizationserver-848b574b44-b7dxx   0/1     ContainerCreating   0          29s
kubeflow      mysql-7d8b8ff4f4-szkkv                             0/1     ContainerCreating   0          29s
kubeflow      workflow-controller-589ff7c479-8cd4s               0/1     ContainerCreating   0          29s
```
To make sure that all the pods have been correctly initialized and running you can run the following command.


You should all the pods in `Running` state.


```bash
$ kubectl get pods -n kubeflow

NAME                                               READY   STATUS    RESTARTS        AGE
cache-deployer-deployment-6f9c8677b-m8r2w          1/1     Running   0               9m48s
cache-server-d86b59fb4-zmnmm                       1/1     Running   0               9m48s
metadata-envoy-deployment-fd499f77f-4wwnd          1/1     Running   0               9m48s
metadata-grpc-deployment-5644fb9768-sksvz          1/1     Running   0               9m47s
metadata-writer-7b54467cd4-47bmm                   1/1     Running   0               9m46s
minio-55464b6ddb-gsnvf                             1/1     Running   0               9m46s
ml-pipeline-6fc946c6d9-8qgfk                       1/1     Running   2 (7m45s ago)   9m46s
ml-pipeline-persistenceagent-5958478546-lf6zn      1/1     Running   0               9m46s
ml-pipeline-scheduledworkflow-5c4dfc67c-d8f2b      1/1     Running   0               9m46s
ml-pipeline-ui-6df94859ff-m7m97                    1/1     Running   0               9m46s
ml-pipeline-viewer-crd-7fccb85dd6-7xnpp            1/1     Running   0               9m45s
ml-pipeline-visualizationserver-848b574b44-b7dxx   1/1     Running   0               9m45s
mysql-7d8b8ff4f4-szkkv                             1/1     Running   0               9m45s
workflow-controller-589ff7c479-8cd4s               1/1     Running   0               9m45s
```
In case you run into problems, you can always look inside a Pod to see what errors or logs is it generated. For example:

```bash

kubectl describe pod pod_name -n kubeflow
```


## 5. Run a simple pipeline

First, you need to expose the Pipeline UI so that you can access it from your browser.

### Expose Pipeline UI

```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```


Visit the address `localhost:8080` in your browser and you should be greeted with your very own installation of Kubeflow Pipelines running on your very own Linode Kubernetes Cluster.

Now, we would like to run a simple Pipeline to make sure that everything is working correctly. Before we do that, we need to do some preparation.

The Pipeline I have selected uses MinIO to store data. That is to say that it needs to connect to MinIO instance running . Now you might ask: "Wait! I didn't install any MinIO ?". You're right. But Kubeflow comes with an installation of MinIO since it requires it to store Pipeline metadata and artifacts (e.g. models, data) generated by Pipeline executions. So essentially, we get MinIO installation for free.

Still, you need to know which IP address is it accessible at.

```bash

kubectl describe svc minio-service
```


Once you have the IP address, head over to `kfp-examples/minio-census-pipeline.py` example and change the values in the file accordingly.


## Troubleshooting

See GitHub issue: [Docker vs. Emissary as driver](https://github.com/kubeflow/pipelines/issues/9119)


## Acknowledgements

- MinIO Blog
- Florian Pach on GitHub
- NetworkChuck on Youtube