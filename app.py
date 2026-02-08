from flask import Flask, render_template, jsonify, request
import numpy as np
import torch
import torch.nn.functional as F
from src.infer import predict

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

        # reshape to 2D image
        img = pixels.reshape((height, width))

        # convert to tensor [1,1,H,W]
        img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()

        # downsample to 28x28 for MNIST model
        img_tensor = F.interpolate(img_tensor, size=(28, 28), mode='bilinear', align_corners=False)

        # normalize to [0,1]
        img_tensor /= 255.0

        prediction = predict(img_tensor)

        #plt.imshow(img_tensor[0,0].cpu(), cmap='gray')
        #plt.show()

        return jsonify({"prediction": prediction})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500
    
if __name__=="__main__":
    app.run(debug=True)