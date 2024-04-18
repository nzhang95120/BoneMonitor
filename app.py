from flask import Flask, request, render_template, redirect, url_for, jsonify
import torch
from torchvision import transforms
import os
from PIL import Image
from werkzeug.utils import secure_filename
from classifier import CNN
import joblib

app = Flask(__name__, static_folder='templates/assets')

model = CNN()
model_path = 'models/classifier.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

model_path = 'models/predictor.pkl'
encoders_path = 'models/encoders/'
model2 = joblib.load(model_path)




encoders = {
    'Gender': joblib.load(encoders_path + 'Gender_encoder.pkl'),
    'Hormonal Changes': joblib.load(encoders_path + 'Hormonal_Changes_encoder.pkl'),
    'Family History': joblib.load(encoders_path + 'Family_History_encoder.pkl'),
    'Race Ethnicity': joblib.load(encoders_path + 'Race_Ethnicity_encoder.pkl'),
    'Body Weight': joblib.load(encoders_path + 'Body_Weight_encoder.pkl'),
    'Calcium Intake': joblib.load(encoders_path + 'Calcium_Intake_encoder.pkl'),
    'Vitamin D Intake': joblib.load(encoders_path + 'Vitamin_D_Intake_encoder.pkl'),
    'Physical Activity': joblib.load(encoders_path + 'Physical_Activity_encoder.pkl'),
    'Smoking': joblib.load(encoders_path + 'Smoking_encoder.pkl'),
    'Alcohol Consumption': joblib.load(encoders_path + 'Alcohol_Consumption_encoder.pkl'),
    'Medical Conditions': joblib.load(encoders_path + 'Medical_Conditions_encoder.pkl'),
    'Medications': joblib.load(encoders_path + 'Medications_encoder.pkl'),
    'Prior Fractures': joblib.load(encoders_path + 'Prior_Fractures_encoder.pkl'),
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/classify')
def classify():
    return render_template('classify.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/reccomendations')
def recommendations():
    return render_template('recommendations.html')

@app.route('/classify', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('classify.html', result="No file part", link=None)
    file = request.files['file']
    if file.filename == '':
        return render_template('classify.html', result="No selected file", link=None)
    if file:
        image = Image.open(file.stream)
        image = transform(image).unsqueeze(0)
        output = model(image)
        _, predicted = torch.max(output, 1)
        if predicted.numpy()[0] == 1:  # Assuming 1 is 'osteoporosis'
            result = 'osteoporosis'
            link = '/recommendations'  # This should be the URL to your recommendations tab or page
            message = "We recommend visiting our recommendations page for further advice."
        else:
            result = 'normal'
            link = None
            message = "No further action needed."
        return render_template('classify.html', result=result, link=link, message=message)
    else:
        return render_template('classify.html', result="Invalid file type", link=None)

@app.route('/predict', methods=['POST'])
def test():
    form_data = request.form

    input_data = []

    age = form_data.get('Age', None)
    
    if age is not None:
        try:
            # Convert age to float or int as required by your model
    
            input_data.append(float(age))
        except ValueError:
            return "Error: Invalid input for 'Age'. Please provide a valid number."
    else:
        return "Error: 'Age' is required for prediction."
    

    for feature, encoder in encoders.items():
        if feature in form_data:
            user_input = form_data[feature]
            if(user_input == "None" or user_input == "none"):
                user_input = 'nan'
            if(user_input == "Abnormal"):
                user_input = 'Postmenopausal'
            if user_input is not None and user_input in encoder.classes_:
                encoded = encoder.transform([user_input])[0]
                input_data.append(encoded)
            else:
            # Use the first class as a default or define a specific default for each feature
                default_value = encoder.transform([encoder.classes_[0]])[0]
                input_data.append(default_value)
        else:

            input_data.append(-1)  # Use -1 or any appropriate value that your model can handle as "unknown"

    # Make prediction
    prediction = model2.predict([input_data])

    # Render the prediction in an HTML page
    return render_template('prediction.html', prediction=prediction[0])





if __name__ == '__main__':
    app.run(debug=True)
