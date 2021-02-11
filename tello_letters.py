#Obtener todas las librerias

from djitellopy import Tello
import cv2
import time
import numpy as np
from keras import backend as K
from keras.models import model_from_json

######################################################################
width = 320  # ANCHO DE LA IMAGEN
height = 240  # ALTURA DE LA IMAGEN
startCounter =1   #  0 PARA VUELO 1 PARA PRUEBAS
######################################################################


def arriba():
    dron.move_up(40)
    time.sleep(5)
def abajo():
    dron.move_down(40)
    time.sleep(5)
def izquierda():
    dron.move_left(40)
    time.sleep(5)
def derecha():
    dron.move_right(40)
    time.sleep(5)
def arriba_medio():
    dron.move_up(20)
    time.sleep(5)
def abajo_medio():
    dron.move_down(20)
    time.sleep(5)
def izquierda_medio():
    dron.move_left(20)
    time.sleep(5)
def derecha_medio():
    dron.move_right(20)
    time.sleep(5)


dicc = {'A': [arriba,derecha_medio,abajo,arriba_medio,izquierda_medio],
'B': [arriba,derecha_medio,abajo_medio,izquierda_medio,derecha_medio,abajo_medio,izquierda_medio],
'C': [izquierda,abajo,derecha],
'D': [arriba,derecha_medio,abajo,izquierda_medio],
'E': [izquierda_medio,abajo_medio,derecha_medio,izquierda_medio,abajo_medio,derecha_medio],
'F': [izquierda_medio,abajo_medio,derecha_medio,izquierda_medio,abajo_medio],
'G': [izquierda_medio,abajo_medio,derecha_medio,arriba_medio,izquierda_medio],
'H': [abajo,arriba_medio,derecha_medio,abajo_medio,arriba],
'I': [derecha,izquierda_medio,abajo,izquierda_medio,derecha],
'J': [abajo,izquierda_medio],
'K': [abajo,arriba_medio,derecha_medio,arriba_medio,izquierda_medio,abajo_medio],
'L': [izquierda_medio, arriba],
'M': [arriba,abajo_medio,derecha_medio,abajo_medio, arriba_medio,derecha_medio,abajo_medio],
'N': [arriba, derecha_medio,abajo,derecha_medio,abajo],
'O': [arriba,derecha,abajo,izquierda],
'P': [arriba,derecha_medio,abajo_medio,izquierda_medio],
'Q': [izquierda,arriba,derecha,abajo,izquierda_medio,arriba_medio,abajo],
'R': [arriba,derecha,abajo_medio,izquierda,derecha_medio,abajo_medio],
'S': [izquierda,abajo_medio,derecha,abajo_medio,izquierda],
'T': [arriba,derecha_medio,izquierda],
'U': [abajo,derecha,arriba],
'V': [derecha_medio,abajo,derecha_medio,arriba,derecha_medio],
'W': [abajo,derecha_medio,arriba_medio,abajo_medio,derecha_medio,arriba],
'X': [abajo,arriba_medio,derecha,arriba_medio,abajo],
'Y': [abajo_medio,derecha_medio,abajo_medio,arriba_medio,derecha_medio,arriba_medio],
'Z': [derecha,abajo,derecha]}



#Empezar cargando el modelo de red neuronal  - modelo de aprendizaje automatico (Machine learning)
json_file = open('Entrenamientos/Modelos entrenados/letras.json', 'r')
json_string = json_file.read()
json_file.close()
modelo_letras = model_from_json(json_string)#cargar la arquitectura
# Cargar pesos 
modelo_letras.load_weights("Entrenamientos/Modelos entrenados//letras.h5")

modelo_letras.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #compilar el modelo con el optimizador correcto - adam

word = ""
i = 0

def getLetter(w, h, points):
    global modelo_letras
    # Ahora queremos tomar todos los puntos, pero en una imagen en blanco, y ejecutarlo a través de nuestra red neuronal

    sample = np.zeros([w, h, 3], dtype=np.uint8)  # obtiene una imagen en blanco

    sample.fill(255)  # lo convierte en blanco

    if len(points) == 0: #nada dibujado
        return ""

    # dibuja los puntos en la imagen
    for pt in points:
        sample = cv2.circle(sample, (h - pt[0], pt[1]), 15, (0, 0, 0), -1)

    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)  # convierte a escala de grises

    ret, thresh = cv2.threshold(gray, 127, 255,
                                cv2.THRESH_BINARY_INV)  # aplica un umbral - básicamente, convierte en blanco todos los colores que pasan de un determinado número
    # y el resto son negros

    # algunos de los puntos pueden no estar conectados - esto asegura que no sean recortados
    kernel = np.ones((15, 15), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    # encuentra un contorno basado en los puntos
    contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Encuentra el índice del contorno más grande
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    x, y, w, h = cv2.boundingRect(cnt)  # obtiene los valores de x, y, anchura y altura de la caja delimitadora

    portion = sample[y:y + h, x:x + w]  # recorta la imagen de muestra para obtener la letra

    flipped = cv2.bitwise_not(portion)  # hace que el fondo sea negro

    flipped = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)  # pasa de 3 canales a 1 canal (escala de grises)

    # al introducir una imagen en la red neuronal, hay que voltearla y girarla 90 grados en sentido contrario a las agujas del reloj
    # esto se debe a que los datos de entrenamiento fueron formateados de esta manera
    flipped = cv2.flip(flipped, 1)
    flipped = cv2.rotate(flipped, cv2.ROTATE_90_COUNTERCLOCKWISE)

    letter = cv2.resize(flipped, (28, 28))  # cambiar el tamaño de la letra a 28 * 28
    letter = letter.reshape((1, 1, 28, 28)).astype("float32") / 255  # reformateando y normalizando

    #predecir
    resultado = modelo_letras.predict(letter)  # obtener la predicción

    resultado = resultado[0]  # obtener el vector de salida

    max = np.where(resultado == np.amax(resultado))  # encuentra el índice del valor máximo en el vector de salida - esto nos dará la letra
    return chr(64 + max[0][0])  # obtiene el índice, le añade 64 ya que A es 1 y A es 65 en ASCII


def startVideo():
    global word
    global me

    points = [] #los puntos de la pantalla que representan la punta del marcador
    adding = False #si está añadiendo puntos actualmente - piense como si el marcador estuviera en la página
    erasing = False #borrador
    #ancho, alto y canales del cuadro de video
    w = 0
    h = 0
    c = 0

    while True: #mantener una transmisión en vivo
        leer_frame = dron.get_leer_frame() #obtener el cuadro actual
        frame = leer_frame.frame
        img = cv2.resize(frame, (width, height))
        resultado = frame.copy() #conseguir una copia del cuadro
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #convertir RGB a HSV (otro formato de color)


        # if word == chr(82):
        #     dron.takeoff()
        #     time.sleep(8)
        #     dron.rotate_clockwise(360)
        #     time.sleep(3)
        #     dron.land()
        #     startCounter = 1

        if word:
                print(i)

        #queremos encontrar todo el naranja en la pantalla; esta parte toma un límite inferior y un límite superior para los colores naranja,
        #y luego encuentra todos los colores intermedios
        lower = np.array([0, 128, 234]) #naranja oscuro en HSV
        upper = np.array([34, 183, 255]) #naranja claro en HSV
        mask = cv2.inRange(image, lower, upper) #obtiene una máscara - básicamente, todo el naranja será blanco, y todo lo demás será negro
        resultado = cv2.bitwise_and(resultado, resultado, mask=mask) #operación bitwise, ahora hace que todo el naranja permanezca naranja, y todo el negro negro

        #encuentra contornos - una curva que une todos los colores de la figura
        cnts = cv2.findContours(mask.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2] #obteniendo contornos

        #toma los contornos y dibuja cajas delimitadoras alrededor de las partes naranjas en la pantalla
        if len(cnts) > 0:
            area = max(cnts, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(area) #x, y, ancho, alto
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) #dibujar un rectángulo verde
            if adding: #si el marcador está en la página, queremos añadirlo a los puntos
                points.append([int((2*x + w)/2), int((2*y+h)/2)])
            elif erasing: #si estas borrando
                e_x = int((2*x + w)/2) #obtiene el punto x del borrador
                e_y = int((2*y+h)/2) #obtiene el punto y
                d = 25 #distance
                points = [x for x in points if not (abs(e_x - x[0]) < d and abs(e_y - x[1]) < d)] #comprensión de la lista: si un punto está dentro de los 25 píxeles
                #de la punta del borrador, no se mantendrá en puntos


        w, h, c = frame.shape #actualizaciones w, h, y c

        for pt in points: #dibuja cada punto como un círculo en la página
            frame = cv2.circle(frame, (pt[0], pt[1]), 15, (0, 255, 0), -1)

        #agregar texto a la pantalla

        frame = cv2.flip(frame, 1) #voltea el marco primero, porque la cámara está reflejada

        frame = cv2.putText(frame, word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (80,127,255) , 2, cv2.LINE_AA)

        cv2.imshow('Escribir en el aire - Drone', frame) #muestra el marco
        # cv2.imshow("Myresultado", img)

        #pulsación de tecla
        k = cv2.waitKey(1)

        if k == ord('a'): #a - levanta el marcador
            adding = not adding
            if adding:
                erasing = False
        if k == ord('b'): #retroceso
            if len(word) > 0: word = word[0:len(word)-1]
        if k == ord('s'): #espacio
            word += " "
        elif k == ord('e'): #a - levanta el marcador
            erasing = not erasing
            if erasing:
                adding = False
        elif k == ord('c'): #limpiar la pagina
            points = []
        elif k == ord('d'): #todo listo
            break
        elif k == ord('q'):
            exit(-1)
        elif k == ord('l'):
            dron.land()


    word = getLetter(w, h, points) #obtener la letra y agregar a la palabra
    print(word)
    for i in dicc[word]:
        print(i)
    startVideo()


# Conectarse a Tello
dron = Tello()
dron.connect()
dron.for_back_velocity = 0
dron.left_right_velocity = 0
dron.up_down_velocity = 0
dron.yaw_velocity = 0
dron.speed = 0

print(dron.get_battery())

dron.streamoff()
dron.streamon()

dron.takeoff()
# cap = cv2.VideoCapture(0) #iniciar una transmisión en vivo
startVideo()
dron.land()
