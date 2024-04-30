from flask import Flask, jsonify, request, render_template
import io
from torchvision import transforms
import requests as re
import torch
import base64
from torchvision.models import resnet18
from model import VGG

import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Update with your model's input size
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                         std = [0.229, 0.224, 0.225])  # Update with your model's normalization
])

# Load the models

vgg_model = torch.load('vgg_tank_classifier.pb')
resnet_model = torch.load('resnet_18_tank_classifier.pb')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/predict/url', methods=['POST'])
def predict_image_url():
    image_url = request.json['image_url']
    
    model_type = "resnet"
    if 'model' in request.json:
        model_type = request.json['model'].lower()
        if model_type not in ['resnet','vgg']:
            return jsonify({'error': 'invalid model, model must be resnet or vgg'}), 400


    if model_type == "resnet":
        model = resnet_model
    else:
        model = vgg_model

    model.eval()


    # Fetch the image from the URL
    headers = {
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36"
    }

    last_four_url_char = image_url[-4:].lower()
    if not (last_four_url_char == '.png' or last_four_url_char == '.jpg'):
        return jsonify({'error': 'Check if image url is valid (ends with jpg or png and url is valid)'}), 401

    response = re.get(image_url, headers=headers)
    if response.status_code == 200:
        image = Image.open(io.BytesIO(response.content))
    else:
        return jsonify({'error': 'Failed to fetch image. Check if image is jpg or png and url is valid'}), 402

    try:
        input_image = transform(image).unsqueeze(0)  # Add batch dimension
    except:
        return jsonify({'error': 'Failed to process image.'}), 403

    # Forward pass through the model
    with torch.no_grad():
        output = model(input_image)

    # for resnet which has output size of 2
    if output.size(1) > 1:
        _, predicted = torch.max(output.data, 1)
        prediction = ["Destroyed", "Not Destroyed"][predicted.item()]
        confidence = round(torch.nn.Softmax(dim = 1)(output).max().item(),3)

        if confidence < 0.65:
            prediction += ' (Uncertain)'

        # pic_hash = display_img(image, prediction, confidence, model_name = "ResNet-18")
    
    # for vgg which has output size of 1
    elif output.size(1) == 1:
        prediction = "Destroyed" if output[0][0] > 0.5 else "Not Destroyed"
        confidence = round(output[0][0].item() if output[0][0] > 0.5 else 1 - output[0][0].item(),3)
        if confidence < 0.65:
            prediction += ' (Uncertain)'
        # pic_hash = display_img(image, prediction, confidence, model_name = "VGG-BN")

    else:
        return jsonify({'error': 'Failed to process image. Check if image is jpg or png and url is valid'}), 400
    
    return jsonify({'model':model_type,'prediction': prediction, 'confidence': confidence}), 200


@app.route('/api/predict/upload', methods=['POST'])
def predict_image_upload():
    image_data = request.json['image_data']
    
    model_type = "resnet"
    if 'model' in request.json:
        model_type = request.json['model'].lower()
        if model_type not in ['resnet','vgg']:
            return jsonify({'error': 'invalid model, model must be resnet or vgg'}), 400


    if model_type == "resnet":
        model = resnet_model
    else:
        model = vgg_model

    model.eval()


    image_data_binary = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_data_binary))

    input_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Forward pass through the model
    with torch.no_grad():
        output = model(input_image)

    # for resnet which has output size of 2
    if output.size(1) > 1:
        _, predicted = torch.max(output.data, 1)
        prediction = ["Destroyed", "Not Destroyed"][predicted.item()]
        confidence = round(torch.nn.Softmax(dim = 1)(output).max().item(),3)

        if confidence < 0.65:
            prediction += ' (Uncertain)'

        # pic_hash = display_img(image, prediction, confidence, model_name = "ResNet-18")
    
    # for vgg which has output size of 1
    elif output.size(1) == 1:
        prediction = "Destroyed" if output[0][0] > 0.5 else "Not Destroyed"
        confidence = round(output[0][0].item() if output[0][0] > 0.5 else 1 - output[0][0].item(),3)
        if confidence < 0.65:
            prediction += ' (Uncertain)'
        # pic_hash = display_img(image, prediction, confidence, model_name = "VGG-BN")

    else:
        return jsonify({'error': 'Failed to process image'}), 400
    
    return jsonify({'model':model_type,'prediction': prediction, 'confidence': confidence}), 200

