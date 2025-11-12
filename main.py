import os

import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input

import numpy as np
import pickle

from tqdm import tqdm
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


filenames=[]

for file in os.listdir("images/images"):
    filenames.append(os.path.join("images/images",file))


print(filenames)
with open("filenames.pkl","wb") as f:
    pickle.dump(filenames,f)

os.makedirs("images_features", exist_ok=True)
for i in tqdm(filenames):
    x=extranct_features(i)
    # print(i,"images_features/{}.npy".format(i.split('images/images')[1].split('.')[0]))
    np.save("images_features/{}.npy".format(i.split('images/images')[1].split('.')[0]),x)









