# 권장 방법: uvicorn과 함께 필요한 모든 라이브러리를 설치
# !pip install "uvicorn[standard]"

from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import torch
import gc
from collections import deque
from ultralytics import YOLO
import mediapipe as mp
from tensorflow.keras.models import load_model

# 모델 로드 및 설정
LABELS = ["범죄", "일상", "쓰러짐"]
LSTM_MODEL_PATH = r"C:\Users\2580j\Downloads\jhy1.h5"
yolo_model = YOLO("yolo11n.pt")
lstm_model = load_model(LSTM_MODEL_PATH)
SEQUENCE_LENGTH = lstm_model.input_shape[1]

app = FastAPI()

# 프레임을 포즈 데이터로 변환하는 클래스
class FrameToPoseArray:
    def __init__(self, frame, bounding_boxes):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.frame = frame
        self.bounding_boxes = bounding_boxes
        self.cropped_images = []
        self.pose_array = []

    def crop_images(self):
        for bbox in self.bounding_boxes:
            cx, cy, w, h = map(int, bbox)
            x1, y1 = cx - w // 2, cy - h // 2
            x2, y2 = cx + w // 2, cy + h // 2
            cropped_image = self.frame[y1:y2, x1:x2]
            self.cropped_images.append(cropped_image)

    def extract_pose_landmarks(self):
        self.pose_array = []
        for cropped_image in self.cropped_images:
            result = self.pose.process(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            if result.pose_landmarks:
                array = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark])
                self.pose_array.append(array)
            else:
                self.pose_array.append(np.zeros((33, 4)))

    def pad_pose_data(self):
        while len(self.pose_array) < 3:
            self.pose_array.append(np.zeros((33, 4)))
        return np.array(self.pose_array[:3])

    def get_pose_data(self):
        self.crop_images()
        self.extract_pose_landmarks()
        self.pad_pose_data()
        return np.concatenate(self.pose_array)

# WebSocket 엔드포인트: 클라이언트로부터 영상 수신 및 예측
@app.websocket("/camera/stream")
async def video_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    sequence_data = deque(maxlen=SEQUENCE_LENGTH)

    try:
        while True:
            # 클라이언트로부터 프레임 데이터 수신
            frame_bytes = await websocket.receive_bytes()
            frame = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

            # YOLO로 사람 감지
            with torch.no_grad():
                results = yolo_model(frame, verbose=False)

            bounding_boxes = [
                r.boxes.xywh[j].cpu().numpy()
                for r in results if 0 in r.boxes.cls.cpu().numpy()
                for j, c in enumerate(r.boxes.cls.cpu().numpy()) if c == 0
            ]

            if bounding_boxes:
                frame_to_pose = FrameToPoseArray(frame, bounding_boxes)
                pose_data = frame_to_pose.get_pose_data()

                if pose_data.shape == (99, 4):
                    sequence_data.append(pose_data)

                if len(sequence_data) == SEQUENCE_LENGTH:
                    d = np.stack(sequence_data, axis=0).reshape(SEQUENCE_LENGTH, -1)
                    d = d[np.newaxis, :, :]
                    lstm_prediction = lstm_model.predict(d, verbose=0)
                    pred_idx = np.argmax(lstm_prediction)
                    prediction_label = LABELS[pred_idx]

                    # 예측 결과를 클라이언트에 전송
                    await websocket.send_text(f"Prediction: {prediction_label}")

            gc.collect()
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"에러 발생: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
