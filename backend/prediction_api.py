import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

def model():
    model = tf.saved_model.load('./model_output')
    return model

def createBoundingBox(image, threshold = 0.5):
  inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
  inputTensor = tf.convert_to_tensor(inputTensor, dtype = tf.uint8)
  inputTensor = inputTensor[tf.newaxis,...]

  model_1 = model()
  print("Model loaded successfully")

  detections = model_1(inputTensor)

  # convert image to numpy array
  bboxs = detections['detection_boxes'][0].numpy()
  classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
  classScores = detections['detection_scores'][0].numpy()

  img_h, img_w, img_c = image.shape

  bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size = 10, iou_threshold=threshold, score_threshold = threshold)

  result = []
  if len(bboxIdx) != 0:
    for i in bboxIdx:
      bbox = tuple(bboxs[i].tolist())
      classConfidence = round(100*classScores[i])
      classIndex = classIndexes[i]

      #displayText = '{}: {}%'.format(classIndex, classConfidence)
      displayText = '{}: {}%'.format("COTS", classConfidence)

      ymin, xmin, ymax, xmax = bbox
      xmin, xmax, ymin, ymax = (xmin * img_w, xmax * img_w, ymin * img_h, ymax * img_h)
      xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

      color = (255, 0, 0)

      cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness = 1)
      cv2.putText(image, displayText, (xmin, ymin-10), cv2.FONT_HERSHEY_PLAIN, 1, color, thickness = 2)

      #line width around the bbox
      lineWidth = min(int((xmax - xmin)*0.2), int((ymax - ymin) * 0.2))

      cv2.line(image, (xmin, ymin), (xmin + lineWidth, ymin), color, thickness = 5)
      cv2.line(image, (xmin, ymin), (xmin, ymin + lineWidth), color, thickness = 5)

      cv2.line(image, (xmax, ymin), (xmax - lineWidth, ymin), color, thickness = 5)
      cv2.line(image, (xmax, ymin), (xmax, ymin + lineWidth), color, thickness = 5)

      cv2.line(image, (xmin, ymax), (xmin + lineWidth, ymax), color, thickness = 5)
      cv2.line(image, (xmin, ymax), (xmin, ymax - lineWidth), color, thickness = 5)

      cv2.line(image, (xmax, ymax), (xmax - lineWidth, ymax), color, thickness = 5)
      cv2.line(image, (xmax, ymax), (xmax, ymax - lineWidth), color, thickness = 5)     


      result.append({
        "class": int(classIndex),
        "class_name": "COTS",
        "bbox": [xmin, xmax, ymin, ymax], 
        "confidence": int(classConfidence)
      }) 
  return result




