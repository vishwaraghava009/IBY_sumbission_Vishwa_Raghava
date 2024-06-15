import cv2
from deepface import DeepFace

def detect_emotions():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'])
            print(analysis['dominant_emotion'])
        except Exception as e:
            print("Error during emotion analysis:", e)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotions()
