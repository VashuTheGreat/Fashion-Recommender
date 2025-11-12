import cv2
import pandas as pd


import os

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.models import load_model
import numpy as np
import pickle



model=load_model("features_extractor.keras")
filenames=pickle.load(open('filenames.pkl', 'rb'))
data=pd.read_csv("images.csv")
neighbors_model=pickle.load(open('neighbors_model.pkl', 'rb'))



def extranct_features(img_path_model):
    img=image.load_img(img_path_model,target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x=model.predict(x,verbose=0)
    x=x.flatten()
    x=x/np.linalg.norm(x)
    return x

def image_link(file):
    im = data[data['filename'] == filenames[file].split("\\")[-1]]['link']
    return im.iloc[0]


def give_path_get_link(path):
    normalised_result = extranct_features(path)
    distances, indices = neighbors_model.kneighbors([normalised_result])
    result=[]
    json_result=[]
    for file in indices[0]:
        result.append(image_link(file))
        f="styles/"+filenames[file].split("\\")[-1].split(".")[0]+".json"
        json_result.append(open(f,"r").read())



    return result,json_result

if __name__=='__main__':

    normalised_result=extranct_features(r"C:\Users\vansh\Projects\E-Commerece-Recommendation\images\images\1163.jpg")
    distances, indices = neighbors_model.kneighbors([normalised_result])

    for file in indices[0]:

        # webbrowser.open(im.iloc[0])
        temp_img=cv2.imread(filenames[file])
        cv2.imshow(filenames[file],temp_img)
        cv2.waitKey(0)
