import numpy as np
import cv2
import pandas as pd
import os
from tqdm import tqdm

IMG_SIZE = 50

DIR = "D:\\JupyterNotebooks\\AgeRaceGender\\fairface\\train" #path to the folder containing training images

def Createfairfacedata(DIR):
    fairfacedata = []
    for img in tqdm(os.listdir(DIR)):
        path = os.path.join(DIR,img)
        image = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        fairfacedata.append((np.array(image),np.array(img)))
    return fairfacedata
def changingGendertoNumbers(value):
    if value == 'Male':
        value = 0
    elif value == 'Female':
        value = 1
    else:
        print(value)
    return value
def changeEAandSEAtoAsian(value):
    if (value == 'East Asian' or value == 'Southeast Asian'):
        value = 'Asian'
    return value

def changeRaceToNumbers(value):
    if value == 'White':
        value = classdict['White']
    elif value == 'Black':
        value = classdict['Black']
    elif value == 'Asian':
        value = classdict['Asian']
    elif value == 'Indian':
        value = classdict['Indian']
    elif value == 'Middle Eastern':
        value = classdict['Middle Eastern']
    elif value == 'Latino_Hispanic':
        value = classdict['Latino_Hispanic']
    else:
        print(value)
    return value

def changeAgeToNumbers(value):
    if value == '0-2':
        value = age_dict['0-2']
    elif value == '3-9':
        value = age_dict['3-9']
    elif value == '10-19':
        value = age_dict['10-19']
    elif value == '20-29':
        value = age_dict['20-29']
    elif value == '30-39':
        value = age_dict['30-39']
    elif value == '40-49':
        value = age_dict['40-49']
    elif value == '50-59':
        value = age_dict['50-59']
    elif value == '60-69':
        value = age_dict['60-69']
    elif value == 'more than 70':
        value = age_dict['more than 70']
    else:
        print(value)
    return value

if __name__ == '__main__':
    if 'finaldf.pkl' in os.listdir('.\\'): # your working directory
        print('Data is already prepared.')
    else:
        fairFaceData = Createfairfacedata(DIR)
        df = pd.DataFrame()
        df['image'] = [fairFaceData[i][0] for i in range(len(fairFaceData))]
        df['file'] = ['train/' + np.array_str(fairFaceData[i][1]) for i in range(len(fairFaceData))]
        FairFaceLabelsDf = pd.read_csv('D:\\JupyterNotebooks\\AgeRaceGender\\fairface\\fairface_label_train.csv')
        finaldf = pd.merge(FairFaceLabelsDf,df,on = 'file')
        finaldf = finaldf[['image','age','gender','race']]
        classdict = {'White':0,'Black':1,'Asian':2,'Indian':3,'Middle Eastern':4,'Latino_Hispanic':5}
        age_dict = {'50-59':6, '30-39':4, '3-9':1, '20-29':3, '40-49':5, '10-19':2, '60-69':7, '0-2':0,
        'more than 70':8}
        finaldf['gender'] = finaldf['gender'].apply(changingGendertoNumbers)
        finaldf['race'] = finaldf['race'].apply(changeEAandSEAtoAsian)
        finaldf['race'] = finaldf['race'].apply(changeRaceToNumbers)
        finaldf['age']=finaldf['age'].apply(changeAgeToNumbers)
        finaldf.to_pickle('finaldf.pkl') # .pkl file is saved in your working directory
        print('Data has be successfully prepared.')


