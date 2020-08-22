import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# # flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
# # flags.DEFINE_string('weights', './checkpoints/yolov4-416',
# #                     'path to weights file')
# # flags.DEFINE_integer('size', 416, 'resize images to')
# # flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
# # flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
# # flags.DEFINE_string('video', './data/focus number plate.ts', 'path to input video')
# # flags.DEFINE_float('iou', 0.45, 'iou threshold')
# # flags.DEFINE_float('score', 0.25, 'score threshold')
# ##############
# framework='tf'
# weights='./checkpoints/yolov4-416'
# size=608
# tiny=True
# model='yolov4'
# video='./data/focus number plate.ts'
# iou=0.45
# score=0.25



# #     # config = ConfigProto()
# #     # config.gpu_options.allow_growth = True
# #     # session = InteractiveSession(config=config)
# #     #STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
# c=0
# a_v=0
# def cal(l):
#     global c
#     global a_v
#     x_c=(l[2]+l[0])/2
#     y_c=(l[3]+l[1])/2
#     if y_c>=360 and y_c<=460:
#         lo=1
#         if a_v!=lo:
#             a_v=1
#             c=c+1
#     else:
#         lo=0
#         if a_v!=lo:
#             a_v=0
#     return c
# input_size = size
# video_path = video

# print("Video from: ", video_path )

# vid = cv2.VideoCapture(video_path)
# width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
# height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
# size = (width, height)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('./data/output_tiny_608_focus number plate.ts', fourcc, 20.0, size)

   
# saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
# infer = saved_model_loaded.signatures['serving_default']

# while True:
#     return_value, frame = vid.read()
#     if return_value:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image = Image.fromarray(frame)
#     else:
#         raise ValueError("No image! Try with another video format")
#     frame_size = frame.shape[:2]
#     image_data = cv2.resize(frame, (input_size, input_size))
#     image_data = image_data / 255.
#     image_data = image_data[np.newaxis, ...].astype(np.float32)
#     prev_time = time.time()
#     batch_data = tf.constant(image_data)
#     pred_bbox = infer(batch_data)
#     #print(pred_bbox)
#     for key, value in pred_bbox.items():
#         boxes = value[:, :, 0:4]
#         #print("boxes:",(boxes))
#         pred_conf = value[:, :, 4:]

#     boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
#         boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
#         scores=tf.reshape(
#             pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
#         max_output_size_per_class=50,
#         max_total_size=50,
#         iou_threshold=iou,
#         score_threshold=score
#     )
#     pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
#     op=int(pred_bbox[3])
#     if op==0:
#         print(0)
#         cv2.line(frame,(0,360),(720,360),(0,0,255),5)
#         cv2.line(frame,(0,460),(720,460),(0,0,255),5)
#         cv2.rectangle(frame,(60,30), (200,70), (0,0,0), -1)
#         cv2.putText(frame, "count:"+str(int(c)), (60,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (70,11,219), 2)
#         frame = np.asarray(frame)
#         cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
#         out.write(frame)
#         cv2.imshow("result", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'): break
#     else:
#         print(1)
#         num_classes = len(classes)
#         image_h, image_w, _ = frame.shape
#         print(image_h)
#         print(image_w)
#         out_boxes, out_scores, out_classes, num_boxes = pred_bbox
#         for i in range(num_boxes[0]):
#             if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
#             coor = out_boxes[0][i]
#             print(coor)
#             coor[0] = int(coor[0] * image_h)
#             coor[2] = int(coor[2] * image_h)
#             coor[1] = int(coor[1] * image_w)
#             coor[3] = int(coor[3] * image_w)
#             print(f"{coor[1]}:{coor[0]}:{coor[3]}:{coor[2]}")
#             x_min,y_min,x_max,y_max=int(coor[1]),int(coor[0]),int(coor[3]),int(coor[2])
#             c=cal([x_min,y_min,x_max,y_max])
#             cv2.rectangle(frame, (x_min,y_min),(x_max,y_max), (51,51,255), 3)
#             cv2.line(frame,(0,360),(720,360),(0,0,255),5)
#             cv2.line(frame,(0,460),(720,460),(0,0,255),5)
#             cv2.rectangle(frame,(60,30), (200,70), (0,0,0), -1)
#             cv2.putText(frame, "count:"+str(int(c)), (60,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (70,11,219), 2)
#             cv2.putText(frame,"centroid",((x_max+x_min)//2,(y_max+y_min)//2),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#             frame = np.asarray(frame)
#             cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
#             out.write(frame)
#             cv2.imshow("result", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'): break
    # print("final:",int(pred_bbox[3]))
    # image = utils.draw_bbox(frame, pred_bbox)
    # curr_time = time.time()
    # exec_time = curr_time - prev_time
    # result = np.asarray(image)
    # info = "time: %.2f ms" %(1000*exec_time)
    # print(info)
    # cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    # result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # out.write(result)
    # cv2.imshow("result", result)
    # if cv2.waitKey(1) & 0xFF == ord('q'): break

    

###############
# import cv2
# vd_path='/home/parameswar/Documents/Anukai solutions/yolov4/yolov4_tiny/data/focus number plate.ts'
# video = cv2.VideoCapture(vd_path) 
# i=1
# while(True):
#     ret, frame = video.read() 
  
#     if ret == True:
#         if i==10:
#             cv2.imwrite("1.png",frame)
#             break
#         else:
#             i=i+1

# video.release() 
# result.release() 
    
# # Closes all the frames 
# cv2.destroyAllWindows() 

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# img=mpimg.imread('1.png')
# imgplot = plt.imshow(img)
# plt.show()




framework='tf'
weights='./checkpoints/yolov4-416'
size=416
tiny=True
model='yolov4'
video='./data/rec-0007_kAA92jt6_ItNK.mp4'
iou=0.45
score=0.25


def detect(image_data):
    prev_time = time.time()
    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        print("boxes:",(boxes))
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
    if op==0:
        print(0)
        iop=0
        cv2.line(frame,(0,360),(720,360),(0,0,255),5)
        cv2.line(frame,(0,460),(720,460),(0,0,255),5)
        cv2.rectangle(frame,(60,30), (200,70), (0,0,0), -1)
        cv2.putText(frame, "count:"+str(int(c)), (60,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (70,11,219), 2)
        frame = np.asarray(frame)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        out.write(frame)
        cv2.imshow("result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    else:
        print(1)
        iop=1
        num_classes = len(classes)
        image_h, image_w, _ = frame.shape
        print(image_h)
        print(image_w)
        out_boxes, out_scores, out_classes, num_boxes = pred_bbox
        for i in range(num_boxes[0]):
            if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
            coor = out_boxes[0][i]
            print(coor)
            coor[0] = int(coor[0] * image_h)
            coor[2] = int(coor[2] * image_h)
            coor[1] = int(coor[1] * image_w)
            coor[3] = int(coor[3] * image_w)
            print(f"{coor[1]}:{coor[0]}:{coor[3]}:{coor[2]}")
            x_min,y_min,x_max,y_max=int(coor[1]),int(coor[0]),int(coor[3]),int(coor[2])
            c=cal([x_min,y_min,x_max,y_max])
            cv2.rectangle(frame, (x_min,y_min),(x_max,y_max), (51,51,255), 3)
            cv2.line(frame,(0,360),(720,360),(0,0,255),5)
            cv2.line(frame,(0,460),(720,460),(0,0,255),5)
            cv2.rectangle(frame,(60,30), (200,70), (0,0,0), -1)
            cv2.putText(frame, "count:"+str(int(c)), (60,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (70,11,219), 2)
            cv2.putText(frame,"centroid",((x_max+x_min)//2,(y_max+y_min)//2),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            frame = np.asarray(frame)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            out.write(frame)
            cv2.imshow("result", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    return iop


c=0
a_v=0
def cal(l):
    global c
    global a_v
    x_c=(l[2]+l[0])/2
    y_c=(l[3]+l[1])/2
    if y_c>=360 and y_c<=460:
        lo=1
        if a_v!=lo:
            a_v=1
            c=c+1
    else:
        lo=0
        if a_v!=lo:
            a_v=0
    return c



input_size = size
video_path = video

print("Video from: ", video_path )

vid = cv2.VideoCapture(video_path)
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./data/output_tiny_rec-0007_kAA92jt6_ItNK.mp4', fourcc, 20.0, size)

   
saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']
while True:
    return_value, frame = vid.read()
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
    else:
        raise ValueError("No image! Try with another video format")
    frame_size = frame.shape[:2]
    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    iop=detect(image_data)
