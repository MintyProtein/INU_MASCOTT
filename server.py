import os
import time
from datetime import datetime
import json
import argparse
import toml
from threading import Thread
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from pyngrok import ngrok  
from flask import Flask, request, jsonify
from src.server_utils import image_to_base64, RequestQueue, save_results, DATETIME_FORMAT, KST, AVG_GENERATION_TIME
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 요청 큐와 결과 저장소
request_queue = RequestQueue()

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
                batch.append(request_queue.dequeue_request(store_in_active=True))
            # 요청 처리
            model_inference(batch)
            
        time.sleep(1) 

def model_inference(batch):
    prompts = []
    for data in batch:
        prompts.append(config['PROMPT_PREFIX'] + data['prompt'] + config['PROMPT_POSTFIX'])
        
    results = pipe(prompt=prompts,
                   num_inference_steps=30,
                   negative_prompts= config['NEG_PROMPT'],
                   width=1024,
                   height=1024, 
                   guidance_scale=7,
                   ).images
    
    for (data, result_img) in zip(batch, results):
        save_results(data, result_img, config)
        request_queue.finish_request(data['req_id'])
        
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    
    # 해당 u_id의 요청이 대기/생성 중이라면, 요청을 거부
    if request_queue.find_by(key='u_id', value=input_data['u_id'])[0] >= 0:
        return jsonify({
            "status": "duplicated", 
            }), 401
        
    req_id, req_time, req_ahead = request_queue.enqueue_request(input_data)
   
    return jsonify({
        "status": "queued",
        "request_id": req_id,
        "request_ahead": req_ahead,
        "eta": AVG_GENERATION_TIME * (req_ahead)
        }), 200

@app.route('/result/<req_id>', methods=['GET'])
def get_result(req_id):
    result_file = os.path.join(config['RESULT_DIR'], "images", f"{req_id}.jpg")
    
    if os.path.exists(result_file):
        b64_img = image_to_base64(result_file)
        return jsonify({
            "status": "completed",
            "b64_img": b64_img,
            "requests_ahead": 0, 
            "eta": 0
            }), 201
    else:
        req_ahead, data = request_queue.find_by(key='req_id', value=req_id)
        if req_ahead == 0:
            time_active = datetime.strptime(data['time_active'], DATETIME_FORMAT).replace(tzinfo=KST)
            time_current = datetime.now(KST)
            elapsed_time = time_current - time_active
            return jsonify({
                "status": "generating",
                "requests_ahead": req_ahead, 
                "eta": AVG_GENERATION_TIME - elapsed_time.total_seconds()
            }), 202
            
        elif req_ahead > 0:
            return jsonify({
                "status": "queued", 
                "requests_ahead": req_ahead,
                "eta": AVG_GENERATION_TIME * req_ahead
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
