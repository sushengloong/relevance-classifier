# relevance-classifier

Training and serving model for classifying whether a sentence is relevant.

[![Run on Google Cloud](https://deploy.cloud.run/button.svg)](https://deploy.cloud.run)

## Setup

Create a python virtual environment and install dependencies.

```sh
$ python3 -m virtualenv myvenv
$ source myvenv/bin/activate
$ pip install -r requirements.txt
```

Load, process and train model.

```sh
$ python training.py
```

Run flask server and access API.

```sh
$ cd app
$ python main.py

# access swagger UI
$ open http://localhost

# access endpoint
$ curl "http://localhost/classify/about%20earthquake"
{"relevant": true, "sentence": "about earthquake"}
```

Alternatively, run flask server in a docker container.

```sh
$ docker run -d -p 8080:80 relevance-classifier # port 8080 on your host OS
```
