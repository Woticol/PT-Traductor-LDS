import cv2
from cvzone.HandTrackingModule import HandDetector  #Este es un paquete de visión por computadora que facilita el procesamiento de imágenes y las funciones de IA. En el núcleo, utiliza las bibliotecas OpenCV y Mediapipe.
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

margen = 20
tamImg = 300    # Tamaño de la imagen cuadrada
conteo = 0    # Conteo de imagenes para guardar (usar el numero de la última imagen registrada)
letra = "Y"    # Letra que se está registrando
carpeta = f"Data_Letras/{letra}"
# Carpeta donde se guardan las imagenes

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    hands, frame = detector.findHands(frame)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]   # Se extraen las coordenadas y el ancho y largo del rectangulo que encierra la mano
        
        frameCrop = frame[y-margen:y+h+margen, x-margen:x+w+margen]   # Se recorta el video original segun las coordenadas del rectangulo
        if frameCrop.size == 0:     # Para los casos en los que la mano se salga del marco de la camara se continua con la siguiente iteración del bucle
            continue
        ancho = frameCrop.shape[1]  # Ancho del video recortado
        alto = frameCrop.shape[0]   # Alto del video recortado
        
        imgBlanca = 255 * np.ones((tamImg, tamImg, 3), dtype=np.uint8)  # Se crea una imagen blanca, dtype=np.uint8 es para que los valores de los pixeles sean enteros sin signo de 8 bits
        
        aspectRatio = alto / ancho   # Relacion de aspecto ([0]→alto, [1]→ancho)
        #print(aspectRatio)
        if aspectRatio > 1:     # Si la forma de la mano es vertical
            proporcion = tamImg / alto
            anchoCalc = min(math.ceil(ancho * proporcion), tamImg)
            frameCropResize = cv2.resize(frameCrop, (anchoCalc, tamImg))    # El alto del video recortado pasa a ser 300, y el ancho crece proporcionalmente
            anchoGap = math.ceil((tamImg - anchoCalc)/2)
            imgBlanca[0:frameCropResize.shape[0], anchoGap:(frameCropResize.shape[1]+anchoGap)] = frameCropResize # Se pega la imagen recortada en la imagen blanca
        elif aspectRatio <= 1:      # Si la forma de la mano es horizontal
            proporcion = tamImg / ancho
            altoCalc = min(math.ceil(alto * proporcion), tamImg)
            frameCropResize = cv2.resize(frameCrop, (tamImg, altoCalc))     # El ancho del video recortado pasa a ser 300, y el alto crece proporcionalmente
            altoGap = math.ceil((tamImg - altoCalc)/2)
            imgBlanca[altoGap:(frameCropResize.shape[0]+altoGap), 0:frameCropResize.shape[1]] = frameCropResize # Se pega la imagen recortada en la imagen blanca
            
        # Guardar imagenes presionando la tecla s
        if cv2.waitKey(1) & 0xFF == ord('s'):
            conteo += 1
            cv2.imwrite(f"{carpeta}/Letra_{letra}_{conteo}.jpg", imgBlanca)
            print(f"Imagen guardada: Letra_{letra}_{conteo}.jpg")
            
        cv2.imshow("Mano", frameCrop)
        cv2.imshow("Blanca", imgBlanca)
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()