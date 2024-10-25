import os
import json
import argparse
import toml
from threading import Thread
import time
import uuid
from collections import deque
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from pyngrok import ngrok  
from flask import Flask, request, jsonify
from src.server_utils import image_to_base64


app = Flask(__name__)

# 요청 큐와 결과 저장소
request_queue = deque()
current_request = []


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", type=str, default="server.toml"
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

def load_model(config, device):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config["BASE_MODEL"],
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.load_lora_weights(config['LORA'])
    pipe.to(device)
    return pipe

def process_batch():
    while True:     
        if len(request_queue) > 0:
            batch = []
            
            # 최대 batch_size 만큼의 요청을 가져옴
            for _ in range(min(config['BATCH_SIZE'], len(request_queue))):
                batch.append(request_queue.popleft())
            # 요청 처리
            model_inference(batch)
            
        time.sleep(1) 

def model_inference(batch):
    prompts = []
    for req_id, input_data, req_time in batch:
        current_request.append(req_id)
        prompts.append(config['PROMPT_PREFIX'] + input_data['prompt'] + config['PROMPT_POSTFIX'])
        
    results = pipe(prompt=prompts,
                   num_inference_steps=30,
                   negative_prompts= config['NEG_PROMPT'],
                   width=1024,
                   height=1024, 
                   guidance_scale=7,
                   ).images
    
    for ((req_id, input_data, req_time), result_img) in zip(batch, results):
        result_path = os.path.join(config['RESULT_DIR'], f"{req_id}.jpg")
        result_img.save(result_path, "JPEG")
        current_request.remove(req_id)
        
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    req_id = str(uuid.uuid4())  # 고유한 요청 ID 생성
    req_time = time.time()
    req_ahead = len(request_queue)
    request_queue.append((req_id, input_data, req_time))
    return jsonify({
        "status": "queued",
        "request_id": req_id,
        "request_ahead": req_ahead,
        "eta": 12 * (req_ahead+1)
        }), 200

@app.route('/result/<req_id>', methods=['GET'])
def get_result(req_id):
    result_file = os.path.join(config['RESULT_DIR'], f"{req_id}.jpg")
    
    if os.path.exists(result_file):
        b64_img = image_to_base64(result_file)
        return jsonify({
            "status": "completed",
            "b64_img": b64_img,
            "requests_ahead": 0, 
            "eta": 0
            }), 201
    elif req_id in current_request:
        return jsonify({
            "status": "generating",
            "requests_ahead": 0, 
            "eta": 12
            }), 202
    else:
        queue_position = None
        for index, (queued_req_id, _, _) in enumerate(request_queue):
            if queued_req_id == req_id:
                queue_position = index
                break
            
        if queue_position is not None:
            return jsonify({
                "status": "queued", 
                "requests_ahead": queue_position,
                "eta": 12 * (queue_position+1)
                }), 203
        
        else:
            return jsonify({"status": "not_found"}), 404
        
        
if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda')
    with open(args.config_file, "r") as file:
        config = toml.load(file)
        print(config)
    pipe = load_model(config, device)
    Thread(target=process_batch).start()
    
    # ngrok으로 공개할 포트 설정 (Flask 기본 포트는 5000)
    public_url = ngrok.connect(addr=config['PORT'], proto="http", hostname=config["NGROK_HOSTNAME"])
    print("Fixed Ngrok Tunnel URL:", public_url)

    # Flask 서버 실행
    app.run(port=config['PORT'])
