from flask import Flask, request, jsonify
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/process_data', methods=['POST'])
def process_data():
    # 解析JSON数据
    print("ok")
    data_dict = request.get_json()
    img_data = np.array(data_dict['image'],dtype=np.uint8)
    pose_info = np.array(data_dict['pose'])


    # model_result = your_ml_model_inference(img, pose_info)
    result = {'result': 'Some result'}

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777)
