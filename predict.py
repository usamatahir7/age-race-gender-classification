import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gdrmodel = keras.models.load_model('./models/gender-cnn-64x3-dense-64x1-1603237140.h5') #path to your saved model
agemodel = keras.models.load_model('./models/age-cnn-64x3-dense-64x1-1603326308.h5') #path to your saved model
racemodel = keras.models.load_model('./models/race-cnn-64x3-dense-64x1-1604142642.h5') #path to your saved model

font                   = cv2.FONT_HERSHEY_COMPLEX
fontScale              = 0.5
fontColor              = (255,255,100)
lineType               = 2
val=input('''Press Enter/Return to start''')
IMG_SIZE = 50


age_dict = {'50-59':6, '30-39':4, '3-9':1, '20-29':3, '40-49':5, '10-19':2, '60-69':7, '0-2':0,
       'more than 70':8}
race_dict = {'White':0,'Black':1,'Asian':2,'Indian':3,'Middle Eastern':4,'Latino_Hispanic':5}
def get_key(val,dict): 
    for key, value in dict.items(): 
        if val == value: 
            return key
      
if val=='':
    #USING WEBCAM:
    video = cv2.VideoCapture(0)

    while True:
        successful_frame_read,frame = video.read()
        gray_scaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rectangle=trained_face_data.detectMultiScale(gray_scaled_img)
        
        gender = 'No Face Detected'
        age = 'N/A'
        gdrCertainty = 'N/A'
        race = 'N/A'


        for i in range (len(face_rectangle)):
            (x,y,w,h)=face_rectangle[i]
            cropped_face=gray_scaled_img[y:y+h,x:x+w]

            cropped_face = cv2.resize(cropped_face,(IMG_SIZE,IMG_SIZE))
            facearray1 = np.array(cropped_face).reshape(-1,IMG_SIZE,IMG_SIZE,1)/255.0
            cropped_face = cv2.blur(cropped_face,(3,3),0)
            
            facearray = np.array(cropped_face).reshape(-1,IMG_SIZE,IMG_SIZE,1)/255.0
            agepred = agemodel.predict(np.array(facearray))
            gdrpred = gdrmodel.predict(np.array(facearray1))
            racepred = racemodel.predict(np.array(facearray))
            if (gdrpred<0.5):
                gender = 'MALE'
                gdrCertainty = round(100 - gdrpred[0][0]*100) 
            else:
                gender = 'FEMALE'
                gdrCertainty = round(gdrpred[0][0]*100) 
        

            race = np.argmax(racepred)
            race = get_key(race,race_dict)
            age = np.argmax(agepred)
            age = get_key(age,age_dict)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.putText(frame,gender, 
        (10,200), 
        font, 
        1,
        fontColor,
        lineType)
        cv2.putText(frame,f'Gender Certainty:{gdrCertainty}', 
        (10,250), 
        font, 
        fontScale,
        fontColor,
        lineType)
        cv2.putText(frame,f'Age:{age}', 
        (10,300), 
        font, 
        1,
        fontColor,
        lineType)
        cv2.putText(frame,f'Race:{race}', 
        (10,350), 
        font, 
        1,
        fontColor,
        lineType)
        cv2.putText(frame,'To quit anytime press q', 
        (30,450), 
        font, 
        0.5,
        (100,100,255),
        lineType)
        
        
        cv2.imshow('Face Detector',cv2.resize(frame,(600,400)))
        key = cv2.waitKey(1)
        if key ==113:
            break