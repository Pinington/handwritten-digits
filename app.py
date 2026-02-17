from flask import Flask, render_template, jsonify, request
import numpy as np
import torch
import torch.nn.functional as F
from src.infer import predict
from src.format import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', result="None")


@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()

        width = data["width"]
        height = data["height"]
        pixels = np.array(data["pixels"], dtype=np.float32)

        result = ""

        # reshape to 2D image
        img = pixels.reshape((height, width))

        connex_composants = divide_image(img)
        for c in connex_composants:
            c = pad_to_square(c)
            img_tensor = torch.tensor(c).unsqueeze(0).unsqueeze(0).float()
            img_tensor = F.interpolate(img_tensor, size=(28, 28), mode='bilinear', align_corners=False)
            prediction = predict(img_tensor)
            result += str(prediction)

        #plt.imshow(img_tensor[0,0].cpu(), cmap='gray')
        #plt.show()

        return jsonify({"prediction": result})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500
    
if __name__=="__main__":
    app.run(debug=True)