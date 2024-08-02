import cv2
import mediapipe as mp
import os
import numpy as np
import time
from datetime import datetime

# Inicializar MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Pasta para armazenar as imagens
data_path = 'evasim/face_recognition/face_database'
if not os.path.exists(data_path):
    os.makedirs(data_path)

# Função para capturar a imagem do usuário
def capture_image():
    cap = cv2.VideoCapture(0)

    start_time = time.time()

    while True:
        if not cap.isOpened():
            print("Erro ao acessar a câmera")
            return None
        
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)
        cv2.imshow('Aguarde 4 segundos', frame)
        
        # Espera 4 segundos antes de capturar a imagem
        if time.time() - start_time >= 4:
            cap.release()
            cv2.destroyAllWindows()
            return frame
        if not ret:
            print("Falha ao capturar imagem")
            return
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
    cap.release()
    cv2.destroyAllWindows()
    return None
    

# Função para detectar rosto
def detect_face(image):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return None
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            return image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
    return None

# Função para salvar a imagem
def save_image(face_image, name):
    filename = os.path.join(data_path, f"{name}.png")
    cv2.imwrite(filename, face_image)

# Função para reconhecer rosto
def recognize_face(face_image):
    face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    for file in os.listdir(data_path):
        if file.endswith(".png"):
            saved_image = cv2.imread(os.path.join(data_path, file), cv2.IMREAD_GRAYSCALE)
            res = cv2.matchTemplate(face_image_gray, saved_image, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > 0.7:  # Ajuste o limiar conforme necessário
                return file.split("_")[0]
    return None

def main():
    # Captura a imagem do usuário
    user_image = capture_image()
    if user_image is not None:
        face_image = detect_face(user_image)
        if face_image is not None:
            recognized_name = recognize_face(face_image)
            if recognized_name:
                print(f"Rosto reconhecido como {recognized_name}")
            else:
                name = input("Digite seu nome: ")
                data = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                arqname = f"{name}_{data}"
                save_image(face_image, arqname)
                print(f"Imagem salva como {arqname}.png")
        else:
            print("Nenhum rosto detectado")
    else:
        print("Erro ao capturar a imagem")

if __name__ == "__main__":
    main()


