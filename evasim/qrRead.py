import cv2
from pyzbar import pyzbar
import numpy as np

def decode_qrcode(frame):
    # Decodificar os QR Codes no frame
    decoded_objects = pyzbar.decode(frame)
    
    qrcode_data = None
    for obj in decoded_objects:
        # Pegar os dados do QR Code
        qrcode_data = obj.data.decode("utf-8")
        break 
    
    return qrcode_data

def main():
    cap = cv2.VideoCapture(0)
    qrcode_data = None

    try:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame,1)

            qrcode_data = decode_qrcode(frame)
            
            if qrcode_data:
                print(f'Dado do QR Code: {qrcode_data}')
                break

            # Mostrar a imagem para facilitar o alinhamento do QR Code
            cv2.imshow("QR Code Scanner", frame)

            # Encerrar se a tecla 'q' for pressionada
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    
    cap.release()
    cv2.destroyAllWindows()
    return qrcode_data

if __name__ == "__main__":
    qrcode_value = main()
