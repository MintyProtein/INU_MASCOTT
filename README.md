# INU_MASCOTT

인천대학교 마스코트 캐릭터 **횃불이**의 이미지를 생성하는 **Stable Diffusion** 모델 활용 Repository입니다.


![example_images](./examples/inu_mascott.png)  

## Installation
필수 라이브러리를 설치합니다:
```bash
git clone https://github.com/MintyProtein/INU_MASCOTT.git
cd INU_MASCOTT
pip install -r requirements.txt
```
## Usage
### Gradio 기반 Web Demo 실행
```bash
python app.py
```
### Flask + Ngrok API 서버 실행
```bash
python server.py
```
모든 요청 로그와 생성된 이미지는 `./server_outputs/` 폴더에 저장됩니다.
