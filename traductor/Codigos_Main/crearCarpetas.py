import numpy as np
import os

ruta = os.path.join("Data_Gestos")
gestos = np.array(["hola", "nombre", "sordo", "lengua de señas", "querer", "tener", "trabajo"])    # Gestos que se intentarán detectar
num_secuencias = 60
frames_por_secuencia = 30

for gesto in gestos:
    for secuencia in range(num_secuencias):
        try:                                                            # Try porque si ya existe el directorio da error
            os.makedirs(os.path.join(ruta, gesto, str(secuencia)))
        except:
            pass