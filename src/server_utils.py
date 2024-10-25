import os
import time
import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        # 이미지를 바이너리 모드로 읽어 base64로 인코딩
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string