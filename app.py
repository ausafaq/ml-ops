from flask import Flask, Response

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/health')
def health_check():
    return Response("", status = 200)

if __name__ == '__main__':
    app.run(debug=True)