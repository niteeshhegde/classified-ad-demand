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

Here is a sample POST request with the request body containing data.json in the current directory:
```buildoutcfg
curl -d "@data.json" -H "Content-Type: application/json" -X POST http://localhost:8501/v1/models/my_saved_model:predict
```
