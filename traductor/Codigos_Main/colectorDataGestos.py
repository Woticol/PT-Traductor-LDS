import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp
import math
from funciones.deteccion import deteccion_mediapipe, dibujar_landmark, extraer_keypoints
from funciones.graficos import circulo_cara

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
colores = [(240,202,36), (214,6,241), (0,0,255), (0,255,0), (255,255,255), (140,0,124), (0,129,0), (0, 255, 255)]   # Celeste(0), Rosado(1), Rojo(2), Verde(3), Blanco(4), Morado(5), Verde oscuro(6), Amarillo (7)

ruta = os.path.join("Data_Gestos")
gestos = np.array(["trabajo"])    # Gestos con los que se va a entrenar (Solamente los nuevos)
# Gestos siguientes: "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "Ñ"
# IMPORTANTE: CREAR LAS CARPETAS PRIMERO con crearCarpetas.py
num_secuencias = 20
frames_por_secuencia = 30
secuencia_actual = 40

minutos = int((num_secuencias*(1.8+1)*gestos.shape[0])//60)
segundos = int((num_secuencias*(1.8+1)*gestos.shape[0]) % 60)
print(f"Tiempo estimado: {minutos} minutos, {segundos} segundos ({minutos}:{segundos})")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)    # 1 → Cámara externa, DSHOW → DirectShow (Windows)
# Resolución de la cámara: inicial (640x480), escalada (960x720)
anchoFinal = 780
altoFinal = 600


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:   # Se configuran los parámetros de la detección
    fase_2 = False      # Fase 2 es dejar de mostrar la guia, y mostrar el reconocimiento
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)    # Voltear la imagen horizontalmente

        frame, resultados = deteccion_mediapipe(frame, holistic)       # Se aplica la detección usando el modelo holistic
        imagen, resultados = deteccion_mediapipe(frame, holistic)
        
        # Dibujar la guía para la posicion de la cara y el cuerpo
        # Cara
        cv2.putText(frame, "CARA", (296, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colores[7], 1, cv2.LINE_AA)
        vertices = circulo_cara((328, 163), 40, 30)
        cv2.polylines(frame, [np.array(vertices)], isClosed=True, color=colores[7], thickness=2)
        # Hombros
        cv2.putText(frame, "HOMBROS", (270, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colores[7], 1, cv2.LINE_AA)
        cv2.line(frame, (235, 266), (424, 266), colores[7], 6)
        # Cadera
        cv2.putText(frame, "CADERA", (283, 442), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colores[7], 1, cv2.LINE_AA)
        cv2.line(frame, (235, 455), (424, 455), colores[7], 6)
        
        dibujar_landmark(frame, resultados)
        
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):   # Salir del programa usando la tecla q
            break
        elif tecla == 13:       # Mostrar el reconocimiento usando la tecla "Enter"
            fase_2 = not fase_2
        
        if fase_2:      # Si se presiona enter, se procede a ejecutar la funcion del reconocimiento
            video = imagen
            for gesto in gestos:
                for secuencia in range(num_secuencias):
                    for num_frame in range(frames_por_secuencia):
                        ret, frame = cap.read()
                        frame = cv2.flip(frame, 1)    # Voltear la imagen horizontalmente
                        imagen, resultados = deteccion_mediapipe(frame, holistic)       # Se aplica la detección usando el modelo holsitic
                        
                        # Dibujar la guía para la posicion de la cara y el cuerpo
                        # Cara
                        vertices = circulo_cara((328, 163), 40, 30)
                        cv2.polylines(imagen, [np.array(vertices)], isClosed=True, color=colores[7], thickness=2)
                        # Hombros
                        cv2.line(imagen, (235, 266), (424, 266), colores[7], 6)
                        # Cadera
                        cv2.line(imagen, (235, 455), (424, 455), colores[7], 6)
                        
                        dibujar_landmark(imagen, resultados)    # Se dibujan los puntos y líneas
                        
                        # Esperar 2 segundos entre secuencias, y ver status
                        if num_frame == 0:
                            cv2.putText(imagen, "Comenzando coleccion", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
                            cv2.putText(imagen, "Colectando datos para <{}>. Video numero {}".format(gesto, secuencia+secuencia_actual), (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                            IMGescalada = cv2.resize(imagen, (anchoFinal, altoFinal))
                            cv2.imshow("Video", IMGescalada)
                            cv2.waitKey(1000)
                        else:
                            cv2.putText(imagen, "Colectando datos para <{}>. Video numero {}".format(gesto, secuencia+secuencia_actual), (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                            cv2.rectangle(imagen, (0, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))-25), (1+(math.ceil(num_frame*(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))/frames_por_secuencia))), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), (0,255,0), -1)
                            IMGescalada = cv2.resize(imagen, (anchoFinal, altoFinal))
                            cv2.imshow("Video", IMGescalada)

                        keypoints = extraer_keypoints(resultados)
                        ruta_npy = os.path.join(ruta, gesto, str(secuencia+secuencia_actual), str(num_frame))
                        np.save(ruta_npy, keypoints)       # EL SEGURO
                
                        # Salir presionando q
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            
                            cap.release()
                            cv2.destroyAllWindows()
                            break
                        
            print("Todos los archivos listos")      # Una vez que terminan los ciclos for
            break
        
        else:
            video = frame
        
        IMGescalada = cv2.resize(video, (anchoFinal, altoFinal))
        cv2.imshow("Video", IMGescalada)
                
    cap.release()
    cv2.destroyAllWindows()
