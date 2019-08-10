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


@app.route('/health')
def health_check():
    return Response("", status = 200)

if __name__ == '__main__':
    app.run(debug=True)