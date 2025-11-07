from flask import Flask, render_template, request
import numpy as np
import re
from CountryWise_Prediction import C_Prediction  

app = Flask(__name__)

# Home page with prediction form
@app.route('/')
def index():
    return render_template('Prediction_Form.html')  # Basic HTML form with country + date input

# POST handler for prediction
@app.route('/C_Predict', methods=['POST'])
def country_predict():
    name = request.form['country_name']
    date = request.form['date']

    try:
        if re.search(r"[0-9]", name):
            msg_s = "Input contains numbers."
            return render_template('Prediction_Form.html', msg_s=msg_s)
        elif re.search(r"[@_!#$%^&*()<>?/\\|}{~:]", name):
            msg_s = "Input contains Special Character."
            return render_template('Prediction_Form.html', msg_s=msg_s)
        else:
            predicted_value = C_Prediction(name, date)
            prediction = str(np.round(predicted_value, 2))
            return render_template('Country_Prediction_Output.html',
                                   country_name=name, prediction=prediction)

    except Exception as e:
        print("Error:", e)
        msg = "Prediction Can't be done."
        return render_template('Prediction_Form.html', msg=msg)

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5005)