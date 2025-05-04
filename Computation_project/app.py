from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import traceback


app = Flask(__name__)

# Load model 
with open('SVM.pkl', 'rb') as f:
    model = pickle.load(f)
 
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f) 


@app.route('/', methods = ['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        label_map = {
            1: "Spruce/Fir",
            2: "Lodgepole Pine",
            3: "Ponderosa Pine",
            4: "Cottonwood/Willow",
            5: "Aspen",
            6: "Douglas-fir",
            7: "Krummholz"
        }

        # Receive JSON data
        data = request.get_json()

        values = data.get('data', [])
        if not isinstance(values, list) or len(values) != 33:
            raise ValueError("Expected a list of 33 numeric values.")
        values = [float(x) for x in values]
        
        array = np.array([values])
        scaled = scaler.transform(array)

        # Predict
        prediction = model.predict(scaled)
        pred_label = int(prediction[0])
        pred_name = label_map.get(pred_label, "Unknown")

        return jsonify({'prediction': pred_name})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)