import os
import time
from datetime import datetime, timedelta, timezone
from flask import request, jsonify
import json
import base64
import time
import uuid
from collections import deque
from PIL import Image

KST = timezone(timedelta(hours=9, minutes=11))
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
AVG_GENERATION_TIME = 7

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        # 이미지를 바이너리 모드로 읽어 base64로 인코딩
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def validate_json(required_fields):
    def decorator(f):
        def wrapper(*args, **kwargs):
            data = request.get_json()
            if not data:
                return jsonify({"status": "error", "message": "Request body must be JSON"}), 400
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return jsonify({"status": "error", "message": missing_fields}), 400
            return f(*args, **kwargs)
        wrapper.__name__ = f.__name__
        return wrapper
    return decorator

def save_results(data: dict, img: Image, config: dict):
    img_path = os.path.join(config['RESULT_DIR'], "images", f"{data['req_id']}.jpg")
    img.save(img_path, "JPEG")
    
    data['image_path'] = img_path
    file_path = os.path.join(config['RESULT_DIR'], f"{datetime.now(KST).strftime('%Y%m%d')}.jsonl")
    data['time_finished'] = datetime.now(KST).strftime(DATETIME_FORMAT)
    with open(file_path, 'a') as file:
        json_line = json.dumps(data)
        file.write(json_line + '\n') 
    return

class RequestQueue(deque):
    def __init__(self, *args, **kwargs):
        super(RequestQueue, self).__init__(*args, **kwargs)
        self._active_requests = []
            
    def enqueue_request(self, data: dict):
        data['req_id'] = str(uuid.uuid4())
        data['time_req'] = datetime.now(KST).strftime(DATETIME_FORMAT)
        
        req_ahead = len(self)
        if self._active_requests:
            req_ahead += 1
            
        self.append(data)
        return data['req_id'], data['time_req'], req_ahead
    
    def dequeue_request(self, store_in_active=True) -> dict:
        """요청 큐의 첫번째 원소를 꺼내어 리턴합니다.

        Args:
            store_in_active (bool, optional): 꺼낸 요청의 데이터를 self._active_requests에 저장할지 결정합니다(탐색에 필요).

        Returns:
            dict: 큐의 첫번째 원소
        """
        item = self.popleft()
        if store_in_active:
            item['time_active'] = datetime.now(KST).strftime(DATETIME_FORMAT)
            self._active_requests.append(item)
        return item
    
    def finish_request(self, req_id):
        for i, data in enumerate(self._active_requests):
            if data['req_id'] == req_id:
                del self._active_requests[i]
                return 1
        return 0
            
    def find_by(self, key, value):
        for data in self._active_requests:
            if data[key] == value:
                return 0, data
            
        for req_ahead, data in enumerate(self):
            if data[key] == value:
                return req_ahead + 1, data
            
        return -1, None