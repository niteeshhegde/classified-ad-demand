## AdDemand Application
### End Points
There 2 end points are exposed :

**/ :** This is the root endpoint which renders the AdChecker UI. It conists of a HTML form and a submit button.

**/api/predict :**  This is the api endpoint for prediction. It takes in a request body with all required fields cleans them, and calls TensorFlow Backend. 
It returns a response containing probability of each class along with mentioning the predicted class.

### Production Instance 
The application is deployed on the Google Cloud App Engine.

It consists of Green unicorn server running on a machine with
2 Gi Memory Limits and 1 Core.

DNS exposed is https://tensorflow-283115.wl.r.appspot.com/

Here is a sample POST request with the request body containing data.json in the current directory:
```buildoutcfg
curl -d "@data.json" -H "Content-Type: application/json" -X POST https://tensorflow-283115.wl.r.appspot.com/api/predict
```
### Running Locally
Requirements : pip2, python3

Install the requirements:
```buildoutcfg
pip install requirements.txt
``` 

Running the application:
```buildoutcfg
python3 classified-ad-demand/app/main.py
``` 
This would run the application on localhost:8080.

### Deploying to production:
**Note:** This deployment is followed by the the TensorFlow serving instance previously deployed in Google Kuberenetes Engine.

[AppEngine Python3 Tutorial](https://cloud.google.com/appengine/docs/standard/python3/quickstart) consists of a Hello World Flask Application deployment.

The following command is run after installing gcloud cli and setting up default project:
```buildoutcfg
gcloud app create --project=[YOUR_PROJECT_ID]
```
This creates a project in app-engine.

To deploy a new instance run:
```buildoutcfg
gcloud app deploy app.yaml --project [YOUR_PROJECT_ID]
```