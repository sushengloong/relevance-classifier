import pickle

from flask import Flask, jsonify

app = Flask("relevance-classifier-service")
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/classify/<sentence>")
def classify(sentence):
    prediction = model.predict([sentence])
    response = {
        'relevant': int(prediction[0]) == 1,
        'sentence': sentence
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
