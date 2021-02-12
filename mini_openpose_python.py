import sys
import cv2
import os
from sys import platform

import os
import glob

# dir_path = 'D:/Escritorio/Dron/openpose/build/examples/tutorial_api_python' #os.path.dirname(os.path.realpath(__file__))
# sys.path.append('usr/local/python');
#
# try:
#         from openpose import *
# except:
#         raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
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
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

def set_params():

        params = dict()
        params["logging_level"] = 3
        params["output_resolution"] = "-1x-1"
        params["net_resolution"] = "-1x368"
        params["model_pose"] = "BODY_25"
        params["alpha_pose"] = 0.6
        params["scale_gap"] = 0.3
        params["scale_number"] = 1
        params["render_threshold"] = 0.05
        # If GPU version is built, and multiple GPUs are available, set the ID here
        params["num_gpu_start"] = 0
        params["disable_blending"] = False
        # Ensure you point to the correct path where models are located
        params["model_folder"] = dir_path + "/../../../models/"
        return params

def main():


        params = set_params()

        #Constructing OpenPose object allocates GPU memory
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()


        # openpose = OpenPose(params)

        #Opening OpenCV stream
        stream = cv2.VideoCapture(0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        counter = -1
        while True:
                counter += 1
                ret,img = stream.read()
                # cv2.imwrite('D:/Escritorio/Dron/DronTodo/images/openimage{}.jpg'.format(counter),img)
                # Output keypoints and the image with the human skeleton blended on it
                # keypoints, output_image = openpose.forward(img, True)
                # Process Image
                # img2 = cv2.imread('lena_caption.png', cv2.COLOR_BGR2RGB)
                datum = op.Datum()
                # imageToProcess = cv2.imread('D:/Escritorio/Dron/DronTodo/images/openimage{}.jpg'.format(counter))
                datum.cvInputData = img #imageToProcess
                opWrapper.emplaceAndPop(op.VectorDatum([datum]))

                # Print the human pose keypoints, i.e., a [#people x #keypoints x 3]-dimensional numpy object with the keypoints of all the people on that image
                # if len(datum.poseKeypoints)>0:
                #         print(counter)
                #         print('Human(s) Pose Estimated!')
                #         print(datum.poseKeypoints)
                # else:
                #         print('No humans detected!')


                # Display the stream
                cv2.putText(datum.cvOutputData,'OpenPose using Python-OpenCV',(20,30), font, 1,(255,255,255),1,cv2.LINE_AA)

                cv2.imshow('Human Pose Estimation',datum.cvOutputData)

                key = cv2.waitKey(1)

                if key==ord('q'):
                    # files = glob.glob('D:/Escritorio/Dron/DronTodo/images')
                    # for f in files:
                    #     os.remove(f)
                    break

        stream.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
        main()
