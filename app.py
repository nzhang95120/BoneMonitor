from flask import Flask, request, render_template, redirect, url_for
import torch
from torchvision import transforms
import os
from PIL import Image
from werkzeug.utils import secure_filename
from classifier import CNN

app = Flask(__name__, static_folder='templates/assets')

model = CNN()
model_path = 'models/classifier.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

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

    


if __name__ == '__main__':
    app.run(debug=True)
