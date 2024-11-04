# sweetHome
Npia Google ML Project


### 실행순서
1. app.py 
    - uvicorn app:app --host 0.0.0.0 --port 8000 --reload
2. goCam.py
    - python goCam.py


### 카메라 동작
- 시작
```
import requests
response = requests.post(f"http://localhost:8000/camera/start")
```
- 종료
```
import requests
response = requests.post(f"http://localhost:8000/camera/stop")
```
### 연결 
http://172.23.229.126:8000