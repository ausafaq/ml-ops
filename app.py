from flask import Flask, Response, request, jsonify
from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/train', methods=['POST'])
def train():
    # get parameters from request
    parameters =  request.get_json()

    # read iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # fit model
    clf = svm.SVC(C=float(parameters['C']),
                  probability=True,
                  random_state=1)
    clf.fit(X, y)

    # persist model
    joblib.dump(clf, 'model.pkl')

    return  jsonify({'accuracy' : round(clf.score(X, y) * 100, 2)})

@app.route('/predict', methods=['POST'])
def predict():
    # make prediction from iris input object
    X = request.get_json()
    X = [[float(X['sepalLength']),
          float(X['sepalWidth']),
          float(X['petalLength']),
          float(X['petalWidth'])]]

    # load model
    clf = joblib.load('model.pkl')
    probabilities = clf.predict_proba(X)

    response = ([{'name': 'Iris-Setosa', 'value': round(probabilities[0,0] * 100, 2)},
                 {'name': 'Iris-Versicolor', 'value': round(probabilities[0,1] * 100, 2)},
                 {'name': 'Iris-Virginica', 'value': round(probabilities[0,2] * 100, 2)}])

    return response


@app.route('/health')
def health_check():
    return Response("", status = 200)

if __name__ == '__main__':
    app.run(debug=True)