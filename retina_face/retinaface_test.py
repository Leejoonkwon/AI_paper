import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from retinaface import RetinaFace
from PIL import Image
import imageio
from tqdm import tqdm
path_dir="D:\\ai_data\study_data\\frame/"
file_list=os.listdir(path_dir)
# print(file_list[0]) #01.png

# print(len(file_list)) #6

file_name_list=[]
for i in range(len(file_list)):
    file_name_list.append(file_list[i].replace(".jpg",""))
    
# print(file_name_list) #['01', '02', '03', '04', '05', '06']


# face detect
K = []
for i in tqdm(file_name_list):
    retinaface2=RetinaFace.extract_faces(img_path=f"D:\\ai_data\study_data\\frame//{i}.jpg",align=True)
    K.append(retinaface2)
# detect image가 복수형일 경우
# for i in range(len(K)):
    
#     for j in range(len(K[i])):
        
#         imageio.imwrite(f'D:\\ai_data\study_data\crop(retinaface)/{i}_{j}.jpg',K[i][j])
#     # print(len(K[i]))

# for i in tqdm(range(len(K))):
#     for j in tqdm(range(len(K[i]))):
#         img = Image.fromarray(K[i][j])
#         img_resize=img.resize((512,512),Image.LANCZOS)
#         img_resize.save(f'D:\\ai_data\study_data\crop(retinaface)/frame{i}_{j}.jpg',K[i][j])

for i in tqdm(range(len(K))):
    
    for j in tqdm(range(len(K[i]))):
        img = Image.fromarray(K[i][j])
        img_resize = img.resize((512,512),Image.LANCZOS)
        imageio.imwrite(f'D:\\ai_data\study_data\crop(retinaface)/retina{i}_{j}.jpg',img_resize)
















