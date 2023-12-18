import cv2
from cvzone.HandTrackingModule import HandDetector  #Este es un paquete de visión por computadora que facilita el procesamiento de imágenes y las funciones de IA. En el núcleo, utiliza las bibliotecas OpenCV y Mediapipe.
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os
from funciones.graficos import grafico_dinamico, grafico_dinamico_ABC

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)    # 1 → Cámara externa, DSHOW → DirectShow (Windows)
# Resolución de la cámara: inicial (640x480); escalada (960x720)
anchoFinal = 960
altoFinal = 720

# Grabar
#salida = cv2.VideoWriter("trailerABC.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720))

detector = HandDetector(maxHands=1)

# Se carga el modelo (keras) y las etiquetas
#ruta = os.path.abspath(__file__)
#directorio = os.path.dirname(ruta)
#ruta_model = os.path.join(directorio, "Modelo", "keras_model.h5")
#ruta_labels = os.path.join(directorio, "Modelo", "labels.txt")
#clasificador = Classifier(ruta_model, ruta_labels)
ruta_model = "Modelos/Modelo 21 letras/keras_model.h5"
ruta_labels = "Modelos/Modelo 21 letras/labels.txt"
clasificador = Classifier(ruta_model, ruta_labels)
letras = ["A","B","C","D","E","F",'H','I','K','L','M','N','O','P','Q','R','T','U','V','W','Y']

colores_grafico_antiguo = [(63, 63, 240), (252, 142, 53), (65, 235, 250)]   # Rojo, azul, y amarillo
colores = [(240,202,36), (214,6,241), (0,0,255), (0,255,0), (255,255,255), (140,0,124), (0,129,0), (0, 255, 255)]   # Celeste(0), Rosado(1), Rojo(2), Verde(3), Blanco(4), Morado(5), Verde oscuro(6), Amarillo (7)
colores_graph = [(240,202,36), (214,6,241), (0,0,255), (0,255,0), (140,0,124), (0,129,0), (0, 255, 255)]
color = (245, 176, 65)      # Naranjo

margen = 20
tamImg = 300    # Tamaño de la imagen cuadrada

# Variables para la deteccion
predicciones = []
frase = []
umbral = 0.9

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)    # Voltear la imagen horizontalmente
    frameFinal = frame.copy()   # Se crea una copia del video original, antes de que se detecten las manos
    hands, frame = detector.findHands(frame)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]   # Se extraen las coordenadas y el ancho y largo del rectangulo que encierra la mano
        
        imgBlanca = 255 * np.ones((tamImg, tamImg, 3), dtype=np.uint8)  # Se crea una imagen blanca
        frameCrop = frame[y-margen:y+h+margen, x-margen:x+w+margen]   # Se recorta el video original segun las coordenadas del rectangulo
        if frameCrop.size == 0:     # Para los casos en los que la mano se salga del marco de la camara se continua con la siguiente iteración del bucle
            continue
        ancho = frameCrop.shape[1]  # Ancho del video recortado
        alto = frameCrop.shape[0]  # Alto del video recortado
        
        aspectRatio = alto / ancho   # Relacion de aspecto ([0]→alto, [1]→ancho)
        #print(aspectRatio)
        if aspectRatio > 1:     # Si la forma de la mano es vertical
            proporcion = tamImg / alto
            anchoCalc = min(math.ceil(ancho * proporcion), tamImg)
            #print(f"anchoCalc: {anchoCalc}")
            frameCropResize = cv2.resize(frameCrop, (anchoCalc, tamImg))    # El alto del video recortado pasa a ser 300, y el ancho crece proporcionalmente
            anchoGap = math.ceil((tamImg - anchoCalc)/2)
            imgBlanca[0:frameCropResize.shape[0], anchoGap:(frameCropResize.shape[1]+anchoGap)] = frameCropResize # Se pega la imagen recortada en la imagen blanca
            predict, index = clasificador.getPrediction(imgBlanca,draw=False)    # Se clasifica la imagen
            print(predict, index)
            predicciones.append(index)
            if len(predicciones) >= 25 and len(np.unique(predicciones[-25:])) == 1 and np.unique(predicciones[-25:])[0] == index:      # Esto es un filtro para que una predicción se mantenga un poco en el tiempo para que pueda ser considerada. El valor de -25 se podria considerar como el "tiempo necesario"
                if predict[index] > umbral:    # Si la predicción supera el umbral de confianza
                    tecla2 = cv2.waitKey(1) & 0xFF
                    if tecla2 == ord('f'):      # Al presionar la tecla f, se genera la frase bien
                        #frase_bien = generar_palabra(frase)     # Hay que crear la funcion
                        frase = []      # Se borra la frase para dar paso a una nueva
                        tecla2 = 1
                    if len(frase) > 0:      # 2. Si la frase ya contiene al menos una palabra:
                        if letras[index] != frase[-1]: # Debe corroborar que la siguiente predicción no sea la misma palabra que la anterior para poder
                                                                    # añadirla a la frase, para evitar que la frase se llene de la misma palabra al estar contantemente detectando
                            frase.append(letras[index])
                    else:                   # 1. Si es la primera palabra, se añade a la frase
                        frase.append(letras[index])
            
        elif aspectRatio <= 1:      # Si la forma de la mano es horizontal
            proporcion = tamImg / ancho
            altoCalc = min(math.ceil(alto * proporcion), tamImg)
            #print(f"altoCalc: {altoCalc}")
            frameCropResize = cv2.resize(frameCrop, (tamImg, altoCalc))     # El ancho del video recortado pasa a ser 300, y el alto crece proporcionalmente
            altoGap = math.ceil((tamImg - altoCalc)/2)
            imgBlanca[altoGap:(frameCropResize.shape[0]+altoGap), 0:frameCropResize.shape[1]] = frameCropResize # Se pega la imagen recortada en la imagen blanca
            predict, index = clasificador.getPrediction(imgBlanca,draw=False)    # Se clasifica la imagen
            print(predict, index)
            predicciones.append(index)
            if len(np.unique(predicciones[-25:])) == 1 and np.unique(predicciones[-25:])[0] == index:      # Esto es un filtro para que una predicción se mantenga un poco en el tiempo para que pueda ser considerada. El valor de -25 se podria considerar como el "tiempo necesario"
                if predict[index] > umbral:    # Si la predicción supera el umbral de confianza
                    tecla2 = cv2.waitKey(1) & 0xFF
                    if tecla2 == ord('f'):      # Al presionar la tecla f, se genera la frase bien
                        #frase_bien = generar_palabra(frase)     # Hay que crear la funcion
                        frase = []      # Se borra la frase para dar paso a una nueva
                        tecla2 = 1
                    if len(frase) > 0:      # 2. Si la frase ya contiene al menos una palabra:
                        if letras[index] != frase[-1]: # Debe corroborar que la siguiente predicción no sea la misma palabra que la anterior para poder
                                                                    # añadirla a la frase, para evitar que la frase se llene de la misma palabra al estar contantemente detectando
                            frase.append(letras[index])
                    else:                   # 1. Si es la primera palabra, se añade a la frase
                        frase.append(letras[index])
        
        # Indicador dinámico
        #cv2.putText(frameFinal, letras[index], (x, y-margen*2), cv2.FONT_HERSHEY_SIMPLEX, 3, colores_graph[index], 4)   # Se escribe la letra clasificada en la copia del video
        #cv2.rectangle(frameFinal, (x-margen,y-margen), (x+w+margen,y+h+margen), colores_graph[index], 4)                       # Se dibuja el rectangulo que encierra la mano en la copia del video

        cv2.putText(frameFinal, letras[index], (x, y-margen*2), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 4)   # Se escribe la letra clasificada en la copia del video
        cv2.rectangle(frameFinal, (x-margen,y-margen), (x+w+margen,y+h+margen), color, 4)   
        
        # Ver el funcionamiento de la detección y clasificación
        #cv2.imshow("Mano", frameCrop)
        #cv2.imshow("Blanca", imgBlanca)
        
    else:                       # Si no detecta manos, se inicializa prediction en 0 para el gráfico
        predict = [0] * len(letras)
    
    # Ver palabra formándose
    cv2.rectangle(frameFinal, (0,0), (640, 40), colores_graph[0], -1)
    cv2.putText(frameFinal, "".join(frase), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (colores[4]), 2, cv2.LINE_AA)
    
    # Gráfico dinámico con función
    frameFinal = grafico_dinamico_ABC(predict, letras, frameFinal)
    
    IMGescalada = cv2.resize(frameFinal, (anchoFinal, altoFinal))
    cv2.imshow("Video", IMGescalada)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#salida.release()
cv2.destroyAllWindows()