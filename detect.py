import tensorflow as tf
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
import time
#flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
#flags.DEFINE_string('weights', './checkpoints/yolov4-416',
  #                  'path to weights file')
flags.DEFINE_integer('size', 608, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_list('images', '1.jpg', 'path to input image')
flags.DEFINE_string('output', './detections/', 'path to output folder')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    images = FLAGS.images

    start = time.process_time()
    saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
    print('Time to load weights: ',time.process_time() - start)
    # loop through images in list and run Yolov4 model on each
    for count, image_path in enumerate(images, 1):
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)
        ###
        infer = saved_model_loaded.signatures['serving_default']
        batch_data = tf.constant(images_data)
        start = time.process_time()
        pred_bbox = infer(batch_data)
        print('Time to infer: ',time.process_time() - start)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        # image = utils.draw_bbox(original_image, pred_bbox)
        # # image = utils.draw_bbox(image_data*255, pred_bbox)
        # image = Image.fromarray(image.astype(np.uint8))
        # image.show()
        # image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        # cv2.imwrite(FLAGS.output + 'detection' + str(count) + '.png', image)
        print(1)
        num_classes = len(classes)
        image_h, image_w, _ = original_image.shape
        #print(image_h)
        #print(image_w)
        out_boxes, out_scores, out_classes, num_boxes = pred_bbox
        for i in range(num_boxes[0]):
            if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
            coor = out_boxes[0][i]
            # print(out_boxes)
            print(coor)
            #print(coor)
            coor[0] = int(coor[0] * image_h)
            coor[2] = int(coor[2] * image_h)
            coor[1] = int(coor[1] * image_w)
            coor[3] = int(coor[3] * image_w)
            #print(f"{coor[1]}:{coor[0]}:{coor[3]}:{coor[2]}")
            x_min,y_min,x_max,y_max=int(coor[1]),int(coor[0]),int(coor[3]),int(coor[2])
            file1 = open("myfile.txt","w")

            s=f"{str(coor[0])} {str(coor[2])} {str(coor[3])}"
            file1.write(s)
            cv2.rectangle(original_image, (x_min,y_min),(x_max,y_max), (51,51,255), 3)
        cv2.imwrite(FLAGS.output + 'detection2' + str(count) + '.png', original_image)
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
