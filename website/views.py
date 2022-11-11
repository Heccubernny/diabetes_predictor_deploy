from flask import Flask, request, render_template
import pickle
import numpy as np

# views: Blueprint = Blueprint('views', __name__)
app = Flask(__name__)
# Load Pickle Model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def make():
    return render_template("predict.html")


@app.route('/predictor', methods=['GET','POST'])
def predict():
    pregnancies = request.form['Pregnancies']
    glucose = request.form['Glucose']
    bloodpressure = request.form['BloodPressure']
    insulin = request.form['Insulin']
    bmi = request.form['BMI']
    age = request.form['Age']

    features = np.array([[pregnancies, glucose, bloodpressure, insulin, bmi, age]], dtype=float)
    prediction = model.predict(features)

    output: float = round(prediction[0], 2)
    # print(prediction)
    if (output == 1):
        output = 'Positive'

    elif(output == 0):
        output = 'Negative'

    else:
        None
    return render_template('predictor.html', data=output)


@app.route('/predictor')
def predictor():
    return render_template("predictor.html")


if __name__ == '__main__':
    app.run(debug = True)
