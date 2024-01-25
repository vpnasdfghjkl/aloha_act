import cv2
import json
import base64
import requests
import numpy as np
url = 'http://localhost:7777/process_data'
def capture_and_send_data():
    '''        
    # ## 801数据收集
    # ### joints:
    # <class 'numpy.ndarray'> (14,)

    # ### cam_followed:
    # <class 'numpy.ndarray'> (480, 640, 3)

    # ### cam_fixed:
    # <class 'numpy.ndarray'> (480, 640, 3)
  
    # ### pid?:
    # <class ?> (?,)
    '''

    cnt = 1
    while cnt > 0:
        cnt -= 1
        image_path = 'cam/image_0.png'
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        cam_followed = np.array(image)
        cam_fixed = np.array(image)
        pose_info = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        data = {
            'cam_followed': cam_followed.tolist(),
            'cam_fixed': cam_fixed.tolist(),
            'pose': pose_info.tolist()
        }

        response = requests.post('http://localhost:7777/process_data', json=data)
        time =requests.get(url=url).elapsed.total_seconds()
        print(time)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            try:
                response_data = response.json()
                target_joints=np.array(response_data)
                target_joints=[round(num,3) for num in target_joints]
                print(target_joints)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {e}")
        else:
            print(f"Request failed with status code: {response.status_code}")

if __name__ == "__main__":
    capture_and_send_data()
