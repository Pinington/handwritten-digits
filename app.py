from flask import Flask, render_template, jsonify, request
import numpy as np
import torch
import torch.nn.functional as F
from src.infer import predict
from src.format import *

import ast
import operator

app = Flask(__name__)

# Allowed operators
operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv
}

def eval_safe(expr: str):
    """
    Safely evaluate a simple arithmetic expression with numbers and +, -, *, /
    Returns the result, or None if the expression is invalid.
    """
    def _eval(node):
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type in operators:
                return operators[op_type](left, right)
            else:
                raise ValueError("Unsupported operator")
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            else:
                raise ValueError("Unsupported constant")
        else:
            raise ValueError("Unsupported expression")
    
    try:
        tree = ast.parse(expr, mode='eval')
        return _eval(tree.body)
    except Exception:
        return None

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
        result = result.replace('[', '(').replace(']', ')')
        return jsonify({"prediction": result, "answer": eval_safe(result)})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500
    
if __name__=="__main__":
    app.run(debug=True)