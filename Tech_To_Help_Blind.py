import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import concurrent.futures
import math
import os
from PIL import Image
from gtts import gTTS
from playsound import playsound
import requests
url='http://192.168.94.187/cam-lo.jpg'
url1='http://192.168.94.187/cam-hi.jpg'
url2='http://192.168.46.58'
im=None
 
def run1():
    cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp=urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        im = cv2.imdecode(imgnp,-1)
        
        cv2.imshow('live transmission',im)
        key=cv2.waitKey(5)
        if key==ord('q'):
            break
            
    cv2.destroyAllWindows()
        
def run2():
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp=urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        im = cv2.imdecode(imgnp,-1)
        bbox, label, conf = cv.detect_common_objects(im)
        if len(bbox)!=0:
          if (160-int(bbox[0][2]))>0:
            playsound("left.mp3")
          elif(bbox[0][0]-160)>0:
              playsound("right.mp3")
          else:
                playsound("right.mp3")
        im = draw_bbox(im, bbox, label, conf)
        
        cv2.imshow('detection',im)
        
        key=cv2.waitKey(5)
        if key==ord('q'):
            break
            
    cv2.destroyAllWindows()
 
def runfr():
    cv2.namedWindow("recognition", cv2.WINDOW_AUTOSIZE)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);
    font = cv2.FONT_HERSHEY_SIMPLEX
    id = 0
    names = ['None','Ayush']
    img_resp=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    img = cv2.imdecode(imgnp,-1)
    minW = 0.1*320
    minH = 0.1*240
    c=1
    e=250
    while True:
        img_resp=urllib.request.urlopen(url1)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        img = cv2.imdecode(imgnp,-1)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
           )
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
            if (100-confidence >20):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
                if (e)%250==0:
                    playsound((str(id)+'.mp3'))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(
                    img, 
                     str(id), 
                    (x+5,y-5),
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )
            cv2.putText(
                    img, 
                    str(confidence), 
                    (x+5,y+h-5), 
                    font, 
                    1, 
                    (255,255,0), 
                    1
                   )  
        cv2.imshow('recognition',img)
        e=e+1
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
def run4(face_id):
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    print("\n [INFO] Initializing face capt1re. Look the camera and wait ...")
    count = 0

    while(True):
        img_resp=urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        img = cv2.imdecode(imgnp,-1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1


            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

            cv2.imshow('image', img)

        key=cv2.waitKey(5)
        if key==ord('q'):
            break
        elif count >= 30:
             break
    print("\n [INFO] Exiting Program and cleanup stuff")
    cv2.destroyAllWindows()
    run5()
    return input("Please Enter their Name")
def run5():
    path = 'dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer/trainer.yml') 
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
def getImagesAndLabels(path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
         if imagePath == 'dataset/.DS_Store':
            continue
         PIL_img = Image.open(imagePath).convert('L')
         img_numpy = np.array(PIL_img,'uint8')
         id = int(os.path.split(imagePath)[-1].split(".")[1])
         faces = detector.detectMultiScale(img_numpy)
         for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids

 
if __name__ == '__main__':
    print("started")
    with concurrent.futures.ProcessPoolExecutor() as executer:
            f2= executer.submit(runfr)
