import pandas as pd
import os
import cv2
import tensorflow as tf
import torch
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def image_return(file_path):
  """
  이미지 파일들의 경로를 받아 파일 내 이미지 리스트 반환

  Args:
    file_path: 이미지 파일들의 경로

  Returns:
    이미지 리스트
  """
  image_list = os.listdir(file_path)
  image_path_list = []
  for i in image_list:
    image_path_list.append(file_path + '/' + i)
  
  image_list = []
  for i in image_path_list:
    image_list.append(cv2.imread(i, cv2.IMREAD_COLOR))
  
  return image_list

def height_width_find(image_list):
  """
  이미지 리스트 내 가장 큰 너비, 높이 값 탐색

  Args:
    image_list: 이미지 리스트

  Returns:
    최대 높이, 너비
  """
  height = int(image_list[0].shape[0])
  width = int(image_list[0].shape[1])

  for i in image_list:
    image_height = int(i.shape[0])
    image_width = int(i.shape[1])
    if height < image_height:
      height = image_height
    if width < image_width:
      width = image_width

  return height, width

def image_transpose(height, width, image_list):
  """
  이미지 크기 변환

  Args:
    height, width : 변환할 이미지의 높이, 너비
    image_list : 변환할 이미지 리스트

  Returns:
    변환된 이미지 리스트
  """
  image_data = []

  for i in image_list:
    image_data.append(tf.image.resize_with_crop_or_pad(i,height, width))

  return image_data

def target_data_make(target_data, normal_data):
  """
  타겟 리스트 생성

  Args:
    target_data : 분류할 데이터
    normal_data : 일반 데이터

  Return:
    라벨 데이터
  """
  return np.concatenate((np.ones(len(target_data)),np.zeros(len(normal_data))))

adenocarcinoma_test_file_path = 'D:/ML/archive/Data/test/adenocarcinoma'
adenocarcinoma_train_file_path = 'D:/ML/archive/Data/train/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib'
adenocarcinoma_valid_file_path = 'D:/ML/archive/Data/valid/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib'

largecell_test_file_path = 'D:/ML/archive/Data/test/large.cell.carcinoma'
largecell_train_file_path = 'D:/ML/archive/Data/train/large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa'
largecell_valid_file_path = 'D:/ML/archive/Data/valid/large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa'

normal_test_file_path = 'D:/ML/archive/Data/test/normal'
normal_train_file_path = 'D:/ML/archive/Data/train/normal'
normal_valid_file_path = 'D:/ML/archive/Data/valid/normal'

squamous_test_file_path = 'D:/ML/archive/Data/test/squamous.cell.carcinoma'
squamous_train_file_path = 'D:/ML/archive/Data/train/squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
squamous_valid_file_path = 'D:/ML/archive/Data/valid/squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'
print(adenocarcinoma_test_file_path)

adenocarcinoma_image_list = image_return(adenocarcinoma_test_file_path) + image_return(adenocarcinoma_train_file_path) + image_return(adenocarcinoma_valid_file_path)
normal_image_list = image_return(normal_test_file_path) + image_return(normal_train_file_path) + image_return(normal_valid_file_path)

print(len(adenocarcinoma_image_list), len(normal_image_list))

image_list = adenocarcinoma_image_list + normal_image_list

height, width = height_width_find(image_list)
height = height//3
width = width//3
print(height, width)

# 이미지 변환
adenocarcinoma_image_list = image_transpose(height, width, adenocarcinoma_image_list)
normal_image_list = image_transpose(height, width, normal_image_list)

# 타겟 데이터 생성
input_data = adenocarcinoma_image_list + normal_image_list
target_data = target_data_make(adenocarcinoma_image_list, normal_image_list)

#훈련, 테스트 세트 분할
from tensorflow import keras
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(input_data, target_data, stratify=target_data)

# 데이터 정규화
train_scaled = np.array(train_input).reshape(-1, height, width, 3)
test_scaled = np.array(test_input).reshape(-1, height, width, 3)

# 모델 생성
model = keras.Sequential()
model.add(keras.layers.Conv2D(16, kernel_size=3, activation='relu', padding='same', input_shape = (height, width,3)))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(50, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(1))

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=10)