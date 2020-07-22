## AdDemand Tensorflow Serving Deployment 
### Running locally
TensorFlow Serving docker image must be pulled into the local image registry.

The artifact is in saved-model format and is present in [Cloud Storage Bucket](https://console.cloud.google.com/storage/browser/dataproc-e3bd1f7b-2e29-4da6-a5c4-077c164fd32a-us-central1/avito%2Fsaved-model%2Fmy_saved%2F?project=skilful-orb-255314).
This must be downloaded to the local machine.

TFServing runs by default on port 8051 and it expects volume mounting of the model in saved-model format at root directory.

Run the following command to deploy locally using docker
```buildoutcfg
docker run -p 8501:8501 --mount type=bind source=<Path to saved model>/my_saved_model target=/models/my_saved_model -e MODEL_NAME=my_saved_model -t tensorflow/serving &
```
Now the tensorflow serving is available locally.

Here is a sample POST request:

```buildoutcfg
curl -d "@data.json" -H "Content-Type: application/json" -X POST http://localhost:8501/v1/models/my_saved_model:predict
```
Note : Request body containing data.json is in the current directory.

### Deploying to GCP
[TensorFlow Serving in Kubernetes](https://www.tensorflow.org/tfx/serving/serving_kubernetes) was reffered to deploy the TF Serving image.

#### Docker Image:
A tensorflow container started in port 8050:
```buildoutcfg
docker run -d --name avito tensorflow/serving
```
The Model is then copied to the root folder of the running container:
```
 docker cp <Path to my saved model>/my_saved_model serving_base:/models/my_saved_model
```
Note: Any further updates to the model can be made by copying a new version in this path.

The environment variables are named and the model is committed:
```buildoutcfg
docker commit --change "ENV MODEL_NAME tf-avito" serving_base $USER/tf_serving
```
As a result new image is now present in the local docker registry with our model.

#### Deploying in Kubernetes:
A new GKE cluster with 2 nodes is created by running the following command:
```buildoutcfg
$ gcloud container clusters create tf-serving-cluster --num-nodes 2
```
Set the default cluster for gcloud container command:
```buildoutcfg
gcloud config set container/cluster tf-serving-cluster
gcloud container clusters get-credentials tf-serving-cluster
```
After logging into cluster docker registry, tag the image with GCP project-name:
```buildoutcfg
docker tag $USER/tf_serving gcr.io/tensorflow-283115/avito
```
Configure Docker:
```buildoutcfg
gcloud auth configure-docker
```
Push the image to GCP Registry:
```buildoutcfg
docker push gcr.io/tensorflow-283115/avito
```
Deploy the application using deployment-app.yaml(in the current directory):
```buildoutcfg
kubectl create -f deployment-app.yaml
```
The application must be deployed in sometime.

#### Production Request/Response
**Request**:

Note : Request body containing data.json is in the current directory.

```buildoutcfg
curl -d "@data.json" -H "Content-Type: application/json" -X POST http://35.199.176.26:8501/v1/models/my_saved_model:predict
```

**Response**:
```buildoutcfg
{
    "predictions": [
        [
            0.988715112,
            3.6933546e-08,
            0.0112848813
        ]
    ]
}
```