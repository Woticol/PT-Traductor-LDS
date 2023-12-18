import cv2
import math
import numpy as np
from unidecode import unidecode

colores = [(240,202,36), (214,6,241), (0,0,255), (0,255,0), (255,255,255), (140,0,124), (0,129,0), (0, 255, 255)]   # Celeste(0), Rosado(1), Rojo(2), Verde(3), Blanco(4), Morado(5), Verde oscuro(6), Amarillo (7)
colores_graph = [(240,202,36), (214,6,241), (0,0,255), (0,255,0), (140,0,124), (0,129,0), (0, 255, 255)]
color = (245, 176, 65)      # Celeste

def circulo_cara(centro, radio, numero_lados):
    # Calcula los vértices del polígono
    vertices = []
    for i in range(numero_lados):
        angulo = i * (360 / numero_lados)
        x = int(centro[0] + radio * math.cos(math.radians(angulo)))
        y = int(centro[1] + 1.3 * radio * math.sin(math.radians(angulo)))
        vertices.append((x, y))
    return vertices

def dibujar_guia(frame):
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
    
    # Escribir instrucciones
    cv2.putText(frame, "Posicionate correctamente,", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colores[7], 1, cv2.LINE_AA)
    cv2.putText(frame, "El traductor no esta", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colores[7], 1, cv2.LINE_AA)
    cv2.putText(frame, "detectando tus movimientos", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colores[7], 1, cv2.LINE_AA)
    cv2.putText(frame, "en este momento.", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colores[7], 1, cv2.LINE_AA)
    cv2.putText(frame, "Selecciona un modo", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colores[7], 1, cv2.LINE_AA)
    cv2.putText(frame, "para continuar", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colores[7], 1, cv2.LINE_AA)
    
    return frame

def grafico_dinamico(predict, gestosASCII, input_frame):
    output_frame = input_frame.copy()

    #for palabra in gestos:
    #    palabraASCII = unidecode(palabra)
    #    gestosASCII.append(palabraASCII)

    for index, prob in enumerate(predict):
        cv2.rectangle(output_frame, (0,60+index*40), (int(prob*100), 90+index*40), color, -1)
        cv2.putText(output_frame, gestosASCII[index], (0, 85+index*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

def grafico_dinamico_ABC(predict, gestos, input_frame):
    output_frame = input_frame.copy()
    for index, prob in enumerate(predict):
        if index < 10:
            cv2.rectangle(output_frame, (0,60+index*40), (int(prob*100), 90+index*40), color, -1)
            cv2.putText(output_frame, gestos[index], (0, 85+index*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        elif index >= 10 and index < 19:
            cv2.rectangle(output_frame, (110,60+(index-10)*40), (110+int(prob*100), 90+(index-10)*40), color, -1)
            cv2.putText(output_frame, gestos[index], (110, 85+(index-10)*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(output_frame, (210,60+(index-19)*40), (210+int(prob*100), 90+(index-19)*40), color, -1)
            cv2.putText(output_frame, gestos[index], (210, 85+(index-19)*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame