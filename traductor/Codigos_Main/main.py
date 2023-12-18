import cv2
import mediapipe as mp
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from funciones.deteccion import deteccion_mediapipe, dibujar_landmark, extraer_keypoints
from funciones.graficos import grafico_dinamico, dibujar_guia
from funciones.generador_texto import generar_frase, generar_tts, borrar_audios
from funciones.letras import detectar_letras
from unidecode import unidecode

gestos = np.array(["hola", "nombre", "sordo", "lengua de señas", "querer", "tener", "trabajo"])    # Gestos que se intentarán detectar
letras = ["A","B","C","D","E","F",'H','I','K','L','M','N','O','P','Q','R','T','U','V','W','Y']

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

secuencia = []      # Para recolectar las secuencias de 30 frames para pasarlas al predict
frase = []
predicciones = []   # Para almacenar un registro de las últimas predicciones
umbral = 0.9        # Confianza minima para mostrar el resultado
frase_bien = ""     # La frase arreglada por GPT

colores = [(240,202,36), (214,6,241), (0,0,255), (0,255,0), (255,255,255), (140,0,124), (0,129,0), (0, 255, 255)]   # Celeste(0), Rosado(1), Rojo(2), Verde(3), Blanco(4), Morado(5), Verde oscuro(6), Amarillo (7)
colores_graph = [(240,202,36), (214,6,241), (0,0,255), (0,255,0), (140,0,124), (0,129,0), (0, 255, 255)]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
anchoFinal = 960
altoFinal = 720

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
model.load_weights("Modelos/modelo_7gestos100.keras")


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:   # Se configuran los parámetros de la detección
    modo_gestos = False      # Modo gestos es dejar de mostrar la guia, y pasar a el reconocimiento de gestos
    modo_abecedario = False  # Modo abecedario es pasar de modo gestos a reconocer letras
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):   # Salir del programa usando la tecla q
            break
        elif tecla == 13:       # Mostrar el reconocimiento usando la tecla "Enter"
            modo_gestos = not modo_gestos
            modo_abecedario = False
        elif tecla == 32:      # Si se presiona ESPACIO, se activa el modo abecedario
            modo_gestos = False
            modo_abecedario = not modo_abecedario
        elif tecla == ord('f') and len(frase) > 1:
            frase_bien = generar_frase(frase)
            print(frase_bien)
            generar_tts(frase_bien)
            frase_bien = unidecode(frase_bien)  # Elimina los acentos
            frase_bien = frase_bien.strip()     # Elimina los espacios en blanco al principio y al final            
            frase_bien = frase_bien.replace("¿", "")    # Elimina los posibles caractéres no soportados
            frase_bien = frase_bien.replace("¡", "")
            frase = []      # Se borra la frase para dar paso a una nueva
        elif tecla == ord('m'):
            frase = ['yo','gustar','playa']
        elif tecla == ord('n'):
            frase = ['tener','perro','bonito','casa']
        elif tecla == 8:
            if frase:
                frase.pop()
        
        if modo_gestos:      # Si estamos en la fase 2, comienza la detección
            imagen, resultados = deteccion_mediapipe(frame, holistic)       # Se aplica la detección usando el modelo holistic
            dibujar_landmark(imagen, resultados)
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
                        if len(frase) > 0:      # 2. Si la frase ya contiene al menos una palabra:
                            if gestos[np.argmax(predict)] != frase[-1]: # Debe corroborar que la siguiente predicción no sea la misma palabra que la anterior para poder
                                                                        # añadirla a la frase, para evitar que la frase se llene de la misma palabra al estar contantemente detectando
                                frase.append(gestos[np.argmax(predict)])
                        else:                   # 1. Si es la primera palabra, se añade a la frase
                            frase.append(gestos[np.argmax(predict)])
                            
                ## Gráfico dinámico
                imagen = grafico_dinamico(predict, gestos, imagen)
            
            video = imagen
            
        elif modo_abecedario:
            fisc = detectar_letras(frame, letras, frase)    # fisc es frameFinal y frase empaquetados
            if fisc is None:        # Para los casos en los que la mano se salga del marco de la camara se devuelve None en la función
                continue            # En esos caso se continua con la siguiente iteración del bucle
            frameFinal, frase = fisc    # Se desempaqueta fisc para obtener los valores que nos interesan
            
            video = frameFinal
        
        else:       # Si no estamos en la fase 2, solo se muestra la guía
            frame = dibujar_guia(frame)
            video = frame
        
        # Ver frase formándose
        cv2.rectangle(video, (0,0), (640, 40), colores[0], -1)
        cv2.putText(video, " ".join(frase), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (colores[4]), 2, cv2.LINE_AA)
        cv2.putText(video, frase_bien, (3,400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (colores[4]), 1, cv2.LINE_AA)

        
        IMGescalada = cv2.resize(video, (anchoFinal, altoFinal))
        cv2.imshow("Video", IMGescalada)
        
    borrar_audios()
    cap.release()
    cv2.destroyAllWindows()