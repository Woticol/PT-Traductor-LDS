import numpy as np
import os
from sklearn.model_selection import train_test_split    # Para separar los datos en train y test
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard      # Monitoreo

# Para abrir Tensorboard:
# Abrir otra terminal y escribir tensorboard --logdir="C:\Users\Byron\Desktop\PROYECTO DE TITULO\LSTM\logs"

ruta = os.path.join("Data_Gestos")
gestos = np.array(["A", "B", "C"])    # Gestos que se intentarán detectar (TODOS)
num_secuencias = 30
frames_por_secuencia = 30
epocas = 3000

minutos = int((epocas*0.05*gestos.shape[0])//60)
segundos = int((epocas*0.05*gestos.shape[0]) % 60)

label_map = {etiqueta:num for num, etiqueta in enumerate(gestos)}     # Crear un diccionario
print("Entrenando para")
print(label_map)
print(f"Tiempo estimado: {minutos} minutos, {segundos} segundos ({minutos}:{segundos})")

secuencias, etiquetas = [], []  # Secuencias = entradas (X). Etiquetas = salidas (Y)
for gesto in gestos:
    for secuencia in range(num_secuencias):
        window = []
        for num_frame in range(frames_por_secuencia):
            res = np.load(os.path.join(ruta, gesto, str(secuencia), "{}.npy".format(num_frame)))
            window.append(res)      # Se arma un paquete llamado window con todos los archivos de 1 gesto
        secuencias.append(window)   # Lista con los datos de cada frame de cada secuencia
        etiquetas.append(label_map[gesto])

print(f"Cargando {np.array(secuencias).shape[0]} videos, de {np.array(secuencias).shape[1]} frames, con {np.array(secuencias).shape[2]} datos cada uno")

X = np.array(secuencias)    # ENTRADA
y = to_categorical(etiquetas).astype(int)   # SALIDA Crea listas de N°gestos elementos que representan cada gesto

# Al final la clave es que la lista secuencia y la lista etiquetas estan vinculadas en su orden. Cada 30 (num_secuencias) secuencias, cambia un indice en la lista etiquetas.
# Esto lo interpeta la función train_test_split() que al crear las variables, deja vinculados los datos con su respectivo gesto.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)   # Los X/y de train estan vinculados y los X/y de test estan vinculados

# Tensorboard
log_dir = os.path.join("logs")
tb_callback = TensorBoard(log_dir=log_dir)

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

# Resumen del modelo
model.summary()

# Entrenamiento del modelo. Este modelo quedará grabado en una instancia en Python hasta que este se cierre
model.fit(X_train, y_train, epochs=epocas, callbacks=[tb_callback])       # EL SEGURO
# En mi pc se demora 0.25 seg cada época. Minutos = (epochs*0.25)/60

# Cargar modelo (en caso de que ya este bien entrenado)
#model.load_weights("LSTM/modelo_5gestos.keras")

# Realizar predicciones y comprobacion
predict = model.predict(X_test)    # Matriz donde cada elemento, contiene a su vez 2 elementos, la prediccion del gesto 1 y la del gesto 2

# Guardar modelo si esta bien entrenado
model.save("Modelos/modelo_abecedario.keras")         # EL SEGURO

# Realizar testeo
acierto = 0
print("Testeo")
print(f"Número de pruebas disponibles: {y_test.shape[0]}")
for i in range(y_test.shape[0]):
    print("~~~~~~~~~~~~~~~~~~")
    print(f"Prueba {i+1}")
    print(f"Entrada: {gestos[np.argmax(predict[i])]}")    # Se usa argmax para obtener el indice que tenga la prediccion mas alta, y ver a cual gesto pertenece ese indice
    print(f"Salida: {gestos[np.argmax(y_test[i])]}")
    if gestos[np.argmax(predict[i])] == gestos[np.argmax(y_test[i])]:
        print("Testeo CORRECTO")
        acierto = acierto+1
    else:
        print("Testeo FALLIDO")
print(f"Porcentaje de acierto: {(acierto/y_test.shape[0])*100}%")

# Evaluar usando Matriz de confusión
yhat = model.predict(X_train)

ytrue = np.argmax(y_train, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print("Matriz de confusión:")
print(multilabel_confusion_matrix(ytrue, yhat))

print(f"Presición de {accuracy_score(ytrue, yhat)*100}%")

print("Hola")