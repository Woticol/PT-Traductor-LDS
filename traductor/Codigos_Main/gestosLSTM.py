import cv2
import numpy as np
import os
import mediapipe as mp
import math
from keras.models import Sequential
from keras.layers import LSTM, Dense
from funciones.deteccion import deteccion_mediapipe, dibujar_landmark, extraer_keypoints
from funciones.graficos import circulo_cara, grafico_dinamico
from funciones.generador_texto import generar_frase

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
colores = [(240,202,36), (214,6,241), (0,0,255), (0,255,0), (255,255,255), (140,0,124), (0,129,0), (0, 255, 255)]   # Celeste(0), Rosado(1), Rojo(2), Verde(3), Blanco(4), Morado(5), Verde oscuro(6), Amarillo (7)
colores_graph = [(240,202,36), (214,6,241), (0,0,255), (0,255,0), (140,0,124), (0,129,0), (0, 255, 255)]

# Modo gestos
gestos = np.array(["bien", "hola", "hoy", "querer", "tener", "trabajo"])    # Gestos que se intentarán detectar
# Modo abecedario
#gestos = np.array(["A", "B", "C"])
num_secuencias = 30
frames_por_secuencia = 30

# Nuevas variables
secuencia = []      # Para recolectar las secuencias de 30 frames para pasarlas al predict
frase = []          # Las detecciones concatenadas
predicciones = []   # Para almacenar un registro de las últimas predicciones
umbral = 0.9        # Confianza minima para mostrar el resultado
frase_bien = ""     # La frase arreglada por GPT

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)    # 1 → Cámara externa, DSHOW → DirectShow (Windows)
# Resolución de la cámara: inicial (640x480); escalada (960x720)
anchoFinal = 960
altoFinal = 720

#cap = cv2.VideoCapture("C:/Users/Byron/Desktop/PROYECTO DE TITULO/Manos ABC/Alfabeto de Lengua de Señas Chilena.mp4")

# Modelo Red Neuronal
model = Sequential() # Construccion del modelo de red neuronal de manera secuencial, capa por capa
# Capas
model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation="relu"))
model.add(LSTM(64, return_sequences=False, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(gestos.shape[0], activation="softmax"))
# Compilación del modelo
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
# Cargar modelo
model.load_weights("Modelos/modelo_6gestos.keras")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:   # Se configuran los parámetros de la detección
    fase_2 = False      # Fase 2 es dejar de mostrar la guia, y mostrar el reconocimiento
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)    # Voltear la imagen horizontalmente

        imagen, resultados = deteccion_mediapipe(frame, holistic)       # Se aplica la detección usando el modelo holistic
        
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
        
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):   # Salir del programa usando la tecla q
            break
        elif tecla == 13:       # Mostrar el reconocimiento usando la tecla "Enter"
            fase_2 = not fase_2
        
        if fase_2:      # Si se presiona enter, se procede a ejecutar la funcion del reconocimiento
            dibujar_landmark(imagen, resultados)       # Alternar para mostrar o no los landmarks
            video = imagen      # Dependiendo de la fase, cambia lo que se muestra
            
            # Predicción
            keypoints = extraer_keypoints(resultados)
            secuencia.append(keypoints)
            secuencia = secuencia[-30:]
            
            if len(secuencia) == 30:        # Cuando la secuencia llegue a sus primeros 30 frames, comienza la predicción
                predict = model.predict(np.expand_dims(secuencia, axis=0))[0]   # Añade una nueva dimension en el eje 0 para hacer coincidir la entrada
                print(predict)
                print(gestos[np.argmax(predict)])   # Mostrar el gesto detectado en la terminal
                predicciones.append(np.argmax(predict))
            
                # Mostrar resultado
                if len(np.unique(predicciones[-25:])) == 1 and np.unique(predicciones[-25:])[0] == np.argmax(predict):      # Esto es un filtro para que una predicción se mantenga un poco en el tiempo para que pueda ser considerada. El valor de -25 se podria considerar como el "tiempo necesario"
                    if predict[np.argmax(predict)] > umbral:    # Si la predicción supera el umbral de confianza
                        tecla2 = cv2.waitKey(1) & 0xFF
                        if tecla2 == ord('f'):      # Al presionar la tecla f, se genera la frase bien
                            frase_bien = generar_frase(frase)
                            frase = []      # Se borra la frase para dar paso a una nueva
                            tecla2 = 1
                        if len(frase) > 0:      # 2. Si la frase ya contiene al menos una palabra:
                            if gestos[np.argmax(predict)] != frase[-1]: # Debe corroborar que la siguiente predicción no sea la misma palabra que la anterior para poder
                                                                        # añadirla a la frase, para evitar que la frase se llene de la misma palabra al estar contantemente detectando
                                frase.append(gestos[np.argmax(predict)])
                        else:                   # 1. Si es la primera palabra, se añade a la frase
                            frase.append(gestos[np.argmax(predict)])
                            
                if len(frase) > 5:      # Si ya se han añadido más de 5 palabras a la frase:
                    frase = frase[-5:]  # Se conservan solo las 5 últimas palabras
                
                # Ver frase formándose
                cv2.rectangle(video, (0,0), (640, 40), colores[0], -1)
                cv2.putText(video, " ".join(frase), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (colores[4]), 2, cv2.LINE_AA)
                cv2.putText(video, frase_bien, (3,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (colores[4]), 2, cv2.LINE_AA)

                # Gráfico dinámico
                video = grafico_dinamico(predict, gestos, video, colores_graph)
                
        else:
            video = frame
        
        IMGescalada = cv2.resize(video, (anchoFinal, altoFinal))
        cv2.imshow("Video", IMGescalada)
        
    cap.release()
    cv2.destroyAllWindows()
