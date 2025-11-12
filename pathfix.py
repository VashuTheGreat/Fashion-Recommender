import pickle
import os
import re




# def convert_path(path):
#     image=path.split("/")[-1]
#
#
#     return image
filenames = os.listdir("images_features")

with open("filenames.pkl","wb") as f:
    pickle.dump(filenames,f)







# file_paths=[os.path.join("images/images",convert_path(f)) for f in filenames]
# print(file_paths)
#
#
#
#
# with open("file_path.pkl","wb") as f:
#     pickle.dump(file_paths,f)


