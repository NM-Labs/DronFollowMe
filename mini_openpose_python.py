import sys
import cv2
import os
from sys import platform
from djitellopy import Tello
import time
from OP import OP, Params

#----------------------------------   Importando OpenPose   -----------------------------------------#
dir_path = 'D:/Escritorio/Dron/openpose/build/examples/tutorial_api_python' # os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python');
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e


# ------------------------------ Funciones ----------------------------------#

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


# ------------------------------ Main ----------------------------------#
def main():

        # dron = Tello()
        # dron.connect()
        # dron.for_back_velocity = 0
        # dron.left_right_velocity = 0
        # dron.up_down_velocity = 0
        # dron.yaw_velocity = 0
        # dron.speed = 0
        # print(dron.get_battery())
        # dron.streamoff()
        # dron.streamon()
        # # dron.takeoff()

        params = Params(dir_path).set_params()
        print(type(params))

        #Constructing OpenPose object allocates GPU memory
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        #Opening OpenCV stream
        stream = cv2.VideoCapture(0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        counter = -1
        while True:
                counter += 1
                print('frame procesed: ',counter)
                ret,img = stream.read()
                print('imagen obtenida')
                img = cv2.resize(img, (320, 240))
                print('imagen rescalada')

                #----Dron----#
                # print('frame procesed: ', counter)
                # leer_frame = dron.get_frame_read()
                # print('imagen obtenida')
                # img = leer_frame.frame
                # print('Imagen extraida')
                # img = cv2.resize(img, (320, 240))
                # print('imagen rescalada')
                #----Fin Dron----#

                datum = op.Datum()
                datum.cvInputData = img #imageToProcess
                print('imagen preparada')
                opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                print('imagen analizada')
                object = OP(datumin=datum)
                print(object.check_pose())


                # Display the stream
                cv2.putText(datum.cvOutputData,'OpenPose using Python-OpenCV',(20,30), font, 1,(255,255,255),1,cv2.LINE_AA)

                cv2.imshow('Human Pose Estimation',datum.cvOutputData)
                print('imagen mostrada')
                time.sleep(1)
                print('tiempo de espera termiando')

                key = cv2.waitKey(1)

                if key==ord('q'):
                    break

        stream.release()
        # dron.land()
        cv2.destroyAllWindows()


if __name__ == '__main__':
        main()
