import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

colores = [(240,202,36), (214,6,241), (0,0,255), (0,255,0), (255,255,255), (140,0,124), (0,129,0), (0, 255, 255)]   # Celeste(0), Rosado(1), Rojo(2), Verde(3), Blanco(4), Morado(5), Verde oscuro(6), Amarillo (7)

def deteccion_mediapipe(imagen, modelo):
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)    # Se convierte la imagen a RGB (del sistema de opencv a mediapipe)
    imagen.flags.writeable = False    # Se desactiva la escritura de la imagen para que sea más eficiente
    resultados = modelo.process(imagen)    # Se procesa la imagen para hacer una predicción
    imagen.flags.writeable = True    # Se activa la escritura de la imagen
    imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)    # Se convierte la imagen a BGR
    return imagen, resultados

def dibujar_landmark(imagen, resultados):   # Se dibujan los puntos de referencia
    mp_drawing.draw_landmarks(imagen, resultados.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=colores[2], thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=colores[2], thickness=1, circle_radius=1))  # Se dibujan los puntos de referencia de la cara
    mp_drawing.draw_landmarks(imagen, resultados.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=colores[2], thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=colores[4], thickness=2, circle_radius=2))  # Se dibujan los puntos de referencia del cuerpo
    mp_drawing.draw_landmarks(imagen, resultados.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=colores[6], thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=colores[3], thickness=2, circle_radius=2))   # Se dibujan los puntos de referencia de la mano izquierda
    mp_drawing.draw_landmarks(imagen, resultados.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=colores[5], thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=colores[1], thickness=2, circle_radius=2))    # Se dibujan los puntos de referencia de la mano derecha

def extraer_keypoints(resultados):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in resultados.pose_landmarks.landmark]).flatten() if resultados.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in resultados.face_landmarks.landmark]).flatten() if resultados.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in resultados.left_hand_landmarks.landmark]).flatten() if resultados.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in resultados.right_hand_landmarks.landmark]).flatten() if resultados.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])     # Regresa una lista de 1662 elementos (33*4 + 468*3 + 21*3 + 21*3)