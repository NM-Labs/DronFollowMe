import sys
import cv2
import os
from sys import platform
from djitellopy import Tello
import time
from OP import OP, Params, Movs
from datetime import date

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

#----------------------------------------------------------------------#
Drone = False
# ------------------------------ Main ----------------------------------#
def main():

        if Drone:
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
            #dron.takeoff()
        else:
            stream = cv2.VideoCapture(0)
            dron = None

        params = Params(dir_path).set_params()
        print(type(params))

        movs = Movs(stream, dron, Drone)
        movements = movs.get_movs()
        print(type(movements))

        #Constructing OpenPose object allocates GPU memory
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        font = cv2.FONT_HERSHEY_SIMPLEX
        counter = -1
        while True:
                counter += 1
                print('frame procesed: ',counter)
                if Drone:
                    leer_frame = dron.get_frame_read()
                    img = leer_frame.frame
                else:
                    ret,img = stream.read()

                print('imagen obtenida')
                img = cv2.resize(img, (320, 240))
                print('imagen rescalada')

                datum = op.Datum()
                datum.cvInputData = img #imageToProcess
                print('imagen preparada')
                opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                print('imagen analizada')
                object = OP(datumin=datum)
                pose = object.check_pose()
                print(pose)

                if pose:
                    movements[pose]
                else:
                    print('No pose idetified!')
                # Display the stream
                cv2.putText(datum.cvOutputData,'OpenPose using Python-OpenCV',(20,30), font, 1,(255,255,255),1,cv2.LINE_AA)

                cv2.imshow('Human Pose Estimation',datum.cvOutputData)
                print('imagen mostrada')
                time.sleep(1)
                print('tiempo de espera termiando')

                key = cv2.waitKey(1)

                if key==ord('q'):
                    break


        if Drone:
            dron.land()
        else:
            stream.release()

        cv2.destroyAllWindows()


if __name__ == '__main__':
        main()
