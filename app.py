from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', result="None")

@app.route('/predict', methods=['POST'])
def predict():
    return render_template('index.html', result="None")

if __name__=="__main__":
    app.run(debug=True)