import cv2
import pandas as pd
from networkx.classes import neighbors
from sympy.physics.quantum.gate import normalized
import webbrowser

import os

import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input

import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
model = ResNet50(weights='imagenet', include_top=False,input_shape=(224,224,3))
model.trainable = False

model=tensorflow.keras.Sequential([
model,
    GlobalAvgPool2D(),
])

def extranct_features(img_path_model):
    img=image.load_img(img_path_model,target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x=model.predict(x,verbose=0)
    x=x.flatten()
    x=x/np.linalg.norm(x)
    return x
import pickle
filenames=pickle.load(open('filenames.pkl', 'rb'))

X=[]
for features_file in os.listdir("images_features"):
    x=np.load("images_features/"+features_file)
    X.append(x)


X=np.array(X)


neighbors_model=NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
neighbors_model.fit(X)

normalised_result=extranct_features(r"C:\Users\vansh\Projects\E-Commerece-Recommendation\images\images\10000.jpg")
distances, indices = neighbors_model.kneighbors([normalised_result])

print(indices)


model.save("features_extractor.keras")
with open("neighbors_model.pkl","wb") as f:
    pickle.dump(neighbors_model,f)



data=pd.read_csv("images.csv")
for file in indices[0]:
    print(filenames[file])
    im = data[data['filename'] == filenames[file].split("\\")[-1]]['link']
    webbrowser.open(im.iloc[0])
    print(im)
    temp_img=cv2.imread(filenames[file])
    cv2.imshow(filenames[file],temp_img)
    cv2.waitKey(0)
