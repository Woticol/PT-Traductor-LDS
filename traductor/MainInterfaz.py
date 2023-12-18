import sys
import cv2
from PyQt5 import uic, QtWidgets, QtGui
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import os
import mediapipe as mp
from funciones.deteccion import deteccion_mediapipe, dibujar_landmark, extraer_keypoints
from funciones.generador_texto import generar_frase, generar_tts, borrar_audios
from keras.models import Sequential
from keras.layers import LSTM, Dense
from funciones.graficos import grafico_dinamico, dibujar_guia
from funciones.letras import detectar_letras
from unidecode import unidecode

gestos = np.array(["hola", "nombre", "sordo", "lengua de señas", "querer", "tener", "trabajo"])    # Gestos que se intentarán detectar
gestosASCII = np.array(["hola", "nombre", "sordo", "lengua de senas", "querer", "tener", "trabajo"])
letras = ["A","B","C","D","E","F",'H','I','K','L','M','N','O','P','Q','R','T','U','V','W','Y']

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

secuencia = []      # Para recolectar las secuencias de 30 frames para pasarlas al predict
frase = []
predicciones = []   # Para almacenar un registro de las últimas predicciones
umbral = 0.9        # Confianza minima para mostrar el resultado
frase_bien = ""     # La frase arreglada por GPT

colores = [(240,202,36), (214,6,241), (0,0,255), (0,255,0), (255,255,255), (140,0,124), (0,129,0), (0, 255, 255)]   # Celeste(0), Rosado(1
colores_graph = [(240,202,36), (214,6,241), (0,0,255), (0,255,0), (140,0,124), (0,129,0), (0, 255, 255)]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
anchoFinal = 780
altoFinal = 600

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
model.load_weights("Modelos\modelo_7gestos100.keras")


class GuiSpot(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        uic.loadUi("ui\gui.ui", self)
        
        self.work_thread = WorkThread()
        
        # Botones

        self.salirv2.clicked.connect(self.salir)
        self.iniciar_guia.clicked.connect(self.mostrar_guia)
        self.bt_fase1.clicked.connect(self.iniciar_fase1)
        self.bt_fase2.clicked.connect(self.iniciar_fase2)
        self.work_thread.Imageupd.connect(self.Imageupd_slot)
        self.bt_traducir.clicked.connect(self.traducir_y_mostrar)
        self.bt_borrar.clicked.connect(self.borrar_palabra)
        self.bt_instru.clicked.connect(self.iniciar_instrucciones)


        self.work_thread.start()
        self.work_thread.set_modo_gestos(False)    
        self.work_thread.set_modo_abecedario(False)

        #Mostrar imagenenes en interfaz
        ruta_imagen3 = "ui/Inacap Logo.png"
        self.label_inacap.setPixmap(QtGui.QPixmap(ruta_imagen3))
        ruta_imagen4 = "ui\webcam.png"
        self.label_wc.setPixmap(QtGui.QPixmap(ruta_imagen4))
        ruta_imagen5 = "ui\pregunta.png"
        self.bt_instru.setIcon(QtGui.QIcon(ruta_imagen5))


    #Funciones
    def salir(self):
        sys.exit()

    def mostrar_guia(self):
        self.work_thread.set_modo_gestos(False)
        self.work_thread.set_modo_abecedario(False)

    def Imageupd_slot(self, Image):
        self.label.setPixmap(QPixmap.fromImage(Image))
    
    def iniciar_fase1(self):
        self.work_thread.set_modo_gestos(True)
        self.work_thread.set_modo_abecedario(False)

    def iniciar_fase2(self):
        self.work_thread.set_modo_abecedario(True)
        self.work_thread.set_modo_gestos(False)

    def Imageupd_slot(self, image):
        self.label_wc.setPixmap(QPixmap.fromImage(image))
        self.lb_td()    
    
    def lb_td(self):
        global frase, frase_bien
        frase_texto = " ".join(frase)
        self.label_tdapi.setText(f"Frase actual: {frase_texto}\nFrase corregida: {frase_bien}")

    def traducir_y_mostrar(self):
        global frase,  frase_bien
        if len(frase) > 1:
            frase_bien = generar_frase(frase)
            print(frase_bien)
            generar_tts(frase_bien)
            frase_bien = unidecode(frase_bien)
            frase_bien = frase_bien.strip()
            frase_bien = frase_bien.replace("¿", "")
            frase_bien = frase_bien.replace("¡", "")
            frase = []
        
    def borrar_palabra(self):
        global frase
        if frase:
            frase.pop()

    def iniciar_instrucciones(self):
        self.ventana3 = Instrucciones()
        self.ventana3.show()

class Instrucciones(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        uic.loadUi("ui\Instrucciones.ui", self)
                  
        # Botones
        self.cerrar_v3.clicked.connect(self.bt_cerrar)

        #Mostrar imagenenes en interfaz
        ruta_imagen3 = "ui/Inacap Logo.png"
        self.label_inacap.setPixmap(QtGui.QPixmap(ruta_imagen3))

        ruta_imagen6 = "ui/abcdario_manos2.png"
        self.lb_senas.setScaledContents(True)
        self.lb_senas.setPixmap(QtGui.QPixmap(ruta_imagen6).scaled(490, 190, aspectRatioMode=Qt.KeepAspectRatio))
        
    def bt_cerrar(self):
        self.close()

class WorkThread(QThread):
  
    Imageupd = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.hilo_corriendo = False
        self.cap = cv2.VideoCapture(0)
        self.modo_gestos = False
        self.modo_abecedario = False

    def set_modo_gestos(self, modo_gestos):
        self.modo_gestos = modo_gestos

    def set_modo_abecedario(self, modo_abecedario):
        self.modo_abecedario = modo_abecedario

    def run(self):
        global frase, frase_bien
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)

                if self.modo_gestos:
                    # Lógica del modo gestos
                    global secuencia
                    imagen, resultados = deteccion_mediapipe(frame, holistic)       # Se aplica la detección usando el modelo holistic
                    #dibujar_landmark(imagen, resultados) #Opcional si desea que aparezca en pantalla los Keypoints
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
                        if len(np.unique(predicciones[-25:])) == 1 and np.unique(predicciones[-25:])[0] == np.argmax(predict):      # Esto es un filtro para que una p
                            if predict[np.argmax(predict)] > umbral:    # Si la predicción supera el umbral de confianza
                                if len(frase) > 0:      # 2. Si la frase ya contiene al menos una palabra:
                                    if gestos[np.argmax(predict)] != frase[-1]: # Debe corroborar que la siguiente predicción no sea la misma palabra que la anterior 
                                                                                # añadirla a la frase, para evitar que la frase se llene de la misma palabra al estar 
                                        frase.append(gestos[np.argmax(predict)])
                                else:                   # 1. Si es la primera palabra, se añade a la frase
                                    frase.append(gestos[np.argmax(predict)])

                        ## Gráfico dinámico
                        imagen = grafico_dinamico(predict, gestosASCII, imagen)

                    video = imagen

                elif self.modo_abecedario:
                    # Lógica del modo abecedario
                    fisc = detectar_letras(frame, letras, frase)
                    if fisc is None:
                        continue
                    frameFinal, frase = fisc
                    video = frameFinal

                else:
                    # Lógica por defecto
                    frame = dibujar_guia(frame)
                    video = frame
                
                if ret:
                    image = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
                    convertir_QT = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                    pic = convertir_QT.scaled(780, 600)
                    self.Imageupd.emit(pic)

            borrar_audios()
            cap.release()
            cv2.destroyAllWindows()

    def stop(self):
        self.hilo_corriendo = False
        self.requestInterruption()
        self.wait()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    GUI = GuiSpot()
    GUI.show()
    sys.exit(app.exec_())
