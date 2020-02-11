import pickle

from flask import Flask
from flask_restplus import Api, Resource

app = Flask(__name__)
api = Api(app)
model = pickle.load(open('model.pkl', 'rb'))


@api.route("/classify/<sentence>")
@api.doc(params={'sentence': 'Sentence for relevance classification'})
class Classifier(Resource):
    def get(self, sentence):
        prediction = model.predict([sentence])
        response = {
            'relevant': int(prediction[0]) == 1,
            'sentence': sentence
        }
        return response


if __name__ == '__main__':
    app.run(port=5000, debug=True)
