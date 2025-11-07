from flask import Flask, render_template, request
from Global_Prediction import G_Prediction
import re

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Global_Prediction.html')

@app.route('/Global_Prediction')
def global_prediction_form():
    return render_template('Global_Prediction.html')

@app.route('/G_Predict', methods=['POST'])
def G_Predict():
    year = request.form['year']
    try:
        # Validate if the input is numeric
        if re.search(r"[^0-9]", year):
            msg_s = "Please enter a valid numeric year only."
            return render_template('Global_Prediction.html', msg_s=msg_s)

        # Debug: print the year received
        print(f"Year received: {year}")

        # Call the prediction function
        predicted_value = G_Prediction(year)

        # Debug: print predicted result
        print(f"Predicted Value: {predicted_value}")

        # Round the prediction to 2 decimal places
        prediction = round(predicted_value, 2)

        # Render output page with results
        return render_template('Global_Prediction_Output.html', year=year, prediction=prediction)

    except Exception as e:
        # Log and handle error
        print("Error in global prediction:", e)
        msg = "Prediction can't be done."
        return render_template('Global_Prediction.html', msg=msg)

if __name__ == '__main__':
    app.run(debug=True, port=5001)