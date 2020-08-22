
import time
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
import shutil, os,glob,random

framework='tf'
weights='./checkpoints/yolov4-416'
size=608
tiny=True
model='yolov4'
# video='E:\\New folder (2)\\tolls_for_labelling\\labelled_videos\\192.168.209.8_014_Lane 14_S20200807080001_E20200807225959.ts'
path="2.jpg"
iou=0.45
score=0.25
f=glob.glob(path)
print(len(f))
print(f[0])
random.shuffle(f)

input_size = size

saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']
# i=1

img_path=path
frame=cv2.imread(img_path)
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image = Image.fromarray(frame)

prev_time = time.time()
frame_size = frame.shape[:2]
image_data = cv2.resize(frame, (input_size, input_size))
save_img=cv2.resize(frame,(608,608))
image_data = image_data / 255.
image_data = image_data[np.newaxis, ...].astype(np.float32)
    #print(image_data.shape)
        
batch_data = tf.constant(image_data)
        
pred_bbox = infer(batch_data)
    #print(pred_bbox)
for key, value in pred_bbox.items():
    boxes = value[:, :, 0:4]
    #print("boxes:",(boxes))
    pred_conf = value[:, :, 4:]

boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou,
        score_threshold=score
)
pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
op=int(pred_bbox[3])

if op==1:
    print(op)
    image_h=608
    image_w=608
    out_boxes, out_scores, out_classes, num_boxes = pred_bbox  #x_min=c_1
    coor = out_boxes[0][0]
    coor[0] = int(coor[0] * image_h)
    coor[2] = int(coor[2] * image_h)
    coor[1] = int(coor[1] * image_w)
    coor[3] = int(coor[3] * image_w)
    x_min,y_min,x_max,y_max=int(coor[1]),int(coor[0]),int(coor[3]),int(coor[2])
    print(x_min)
    cropeed_img=save_img[y_min:y_max, x_min:x_max]
    cv2.imwrite("test_np.jpg",cropeed_img)
        # x=coor[1]+(coor[3]-coor[1])/2python 
        # y=coor[0]+(coor[2]-coor[0])/2
        # width=coor[3]-coor[1]
        # height=coor[2]-coor[0]
        # cla=0
        # file1=open("W:\\param's direc\\inf_toll_labelled\\pre_but_not_empty_np_new\\labelled\\labelled"+str(labelled)+".txt","w")
        # s=f"{str(cla)} {str(x)} {str(y)} {str(width)} {str(height)}"
        # file1.write(s)
        # cv2.imwrite("W:\\param's direc\\inf_toll_labelled\\pre_but_not_empty_np_new\\labelled\\labelled"+str(labelled)+".jpg",save_img)
        # print("nps :",i)
        # labelled=labelled+1
else:
    print(op)
    print("no nump")
        # cv2.imwrite(f"W:\\param's direc\\inf_toll_labelled\\pre_but_not_empty_np_new\\need_to_label\\need_to_label"+str(need_to_labelled)+".jpg",save_img)
        # # print("no_nps :",i)
        # need_to_labelled=need_to_labelled+1
        

curr_time = time.time()
exec_time = curr_time - prev_time
info = "time: %.2f ms" %(1000*exec_time)
print(info)
