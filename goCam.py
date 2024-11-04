import asyncio
import websockets
import cv2
import numpy as np

# 서버에 연결할 WebSocket URL (FastAPI 서버 주소와 일치해야 합니다)
WEBSOCKET_URL = "ws://localhost:8000/camera/stream"

async def send_video_stream():
    async with websockets.connect(WEBSOCKET_URL) as websocket:
        cap = cv2.VideoCapture(0)  # 웹캠 사용

        if not cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("프레임을 읽을 수 없습니다.")
                    break

                # 프레임을 JPEG로 인코딩
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = np.array(buffer).tobytes()

                # 프레임 전송
                await websocket.send(frame_bytes)

                # 서버로부터 예측 결과 수신
                prediction = await websocket.recv()
                print(f"서버 예측: {prediction}")

                # 클라이언트 화면에 프레임 표시
                cv2.imshow("Client Webcam", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"에러 발생: {e}")

        finally:
            cap.release()
            cv2.destroyAllWindows()

# asyncio를 사용해 WebSocket 연결 실행
if __name__ == "__main__":
    asyncio.run(send_video_stream())
