import glob
import imagesize
import os
import json
from PIL import Image
from tqdm import tqdm
import argparse

def convert(dir_data):

  train_data = dir_data + '/VisDrone2019-DET-train/'
  val_data = dir_data + '/VisDrone2019-DET-val/'
  test_data = dir_data + '/VisDrone2019-DET-test-dev/'
  loops = [train_data, val_data, test_data]
  for l in loops:
      print('Solving ', l)
      dict_coco = {}

      dir_imgs = './images/'

      ''' Key: images '''
      print('Solving images')
      dict_image_and_id = {}
      dict_coco['images'] = []
      img_id = 0
      for img in tqdm(glob.glob(l + dir_imgs + '*')):
          # image = Image.open(img)
          width, height = imagesize.get(img)
          # file_name = os.path.split(img)
          file_name_save = os.path.split(img)[-1]
          dict_coco['images'].append({
              "id" : img_id,
              "license" : 1,
              "height" : height,
              "width" : width,
              "file_name": file_name_save
          })
          dict_image_and_id[file_name_save] = img_id
          img_id = img_id + 1

      ''' Key: annotations '''
      print('Solving annotations')
      dir_labels = '/annotations/'
      dict_coco['annotations'] = []
      anno_id = 0
      for file_txt in tqdm(glob.glob(l + dir_labels + '*.txt')):
          # with open(file_txt, 'r') as f:
          #     annotations = f.read()
          annotations = open(file_txt,'r').read()
          annotations = annotations.split('\n')
          for i in range(0, len(annotations)):
              annotations[i] = annotations[i].split(',')
          
          annotations = annotations[:-1]
          for detection in annotations:
              category_id = int(detection[5])
              bbox = [int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])]
              area = int(detection[2]) * int(detection[3])
              segmentation = []
              iscrowd = 0
              img_name = os.path.splitext(os.path.split(file_txt)[-1])[0] + '.jpg'
              image_id = dict_image_and_id[img_name]

              dict_coco['annotations'].append({
              "id": anno_id,
                  "image_id": image_id,
                  "category_id": category_id,
                  "bbox": bbox,
                  "area": area,
                  "iscrowd": 0,
                  "ignore": 0
              })

              anno_id = anno_id + 1

      ''' Key: categories '''

      '''
      pedestrian (1), people (2), bicycle (3), car (4), van (5), 
      truck (6), tricycle (7), awning-tricycle (8), bus (9), motor (10), others (11)
      '''
      dict_coco['categories'] = [{
          "id": 1,
          "name": "pedestrian",
          "supercategory": "none"},
          {
          "id": 2,
          "name": "people",
          "supercategory": "none"},
          {
          "id": 3,
          "name": "bicycle",
          "supercategory": "none"},
          {
          "id": 4,
          "name": "car",
          "supercategory": "none"},
          {
          "id": 5,
          "name": "van",
          "supercategory": "none"},
          {
          "id": 6,
          "name": "truck",
          "supercategory": "none"},
          {
          "id": 7,
          "name": "tricycle",
          "supercategory": "none"},
          {
          "id": 8,
          "name": "awning-tricycle",
          "supercategory": "none"},
          {
          "id": 9,
          "name": "bus",
          "supercategory": "none"},
          {
          "id": 10,
          "name": "motor",
          "supercategory": "none"},
          {
          "id": 11,
          "name": "others",
          "supercategory": "none"}
          ]

      with open('annotations_VisDrone_' + l.split('-')[-1][:-1] + '.json', 'w') as f:
          json.dump(dict_coco, f)

def get_args():
    parser = argparse.ArgumentParser('Train')
    parser.add_argument('--data_dir', type=str, default='./',
                        help='Data dir', dest='data_dir')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    convert(args.data_dir)
