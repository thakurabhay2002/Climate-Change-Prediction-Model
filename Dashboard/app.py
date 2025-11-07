from flask import Flask, render_template
import nbformat
from nbconvert import PythonExporter
import io
import runpy

app = Flask(__name__)

@app.route('/')
def index():
    # Convert .ipynb to executable Python code
    with open("Climate_Change_Prediction-main/Sea_level_prediction.ipynb") as f:
        notebook = nbformat.read(f, as_version=4)
        exporter = PythonExporter()
        python_code, _ = exporter.from_notebook_node(notebook)
    
    # Run the extracted code
    exec(python_code, globals())

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
