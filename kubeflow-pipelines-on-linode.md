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

These commands install the necessary Pods and Services as well as other Kubernetes objects that constitute Kubeflow. Node that all the objects are installed in the `kubeflow` namespace.

Since this process involves downloading and starting up a lot of container images, this can take a while. In my experience, it can take anywhere from 10 to 15 minutes before you Kubeflow installation is up and running. 

To make sure that all the pods have been correctly initialized and running you can run the following command.


You should all the pods in `Running` state.


In case you run into problems, you can always look inside a Pod to see what errors or logs is it generated. For example:

```bash

kubectl describe pod pod_name -n kubeflow
```

## 5. Run a simple pipeline


## Troubleshooting

See GitHub issue: [Docker vs. Emissary as driver](https://github.com/kubeflow/pipelines/issues/9119)


## Acknowledgements

