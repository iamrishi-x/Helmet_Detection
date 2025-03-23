import numpy as np
import time
import cv2
import os
import pyttsx3

def load_yolo():
    labelsPath = os.path.sep.join([YOLO_DIR, "cocohelmet.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    
    weightsPath = os.path.sep.join([YOLO_DIR, "yolov3-obj_2400.weights"])
    configPath = os.path.sep.join([YOLO_DIR, "yolov3-obj.cfg"])
    
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    
    layer_names = net.getLayerNames()
    try:
        ln = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    except AttributeError:
        ln = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    return LABELS, COLORS, net, ln

def detect_helmet(image, LABELS, COLORS, net, ln):
    (H, W) = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    boxes, confidences, classIDs = [], [], []
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Configurations
YOLO_DIR = "yolo-coco"
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3
engine = pyttsx3.init()

LABELS, COLORS, net, ln = load_yolo()

def process_video(video_path, output_path):
    vs = cv2.VideoCapture(video_path)
    writer = None
    fps = vs.get(cv2.CAP_PROP_FPS)
    total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while True:
        ret, frame = vs.read()
        if not ret:
            break

        frame = detect_helmet(frame, LABELS, COLORS, net, ln)
        
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1], frame.shape[0]), True)
        writer.write(frame)
        
        current_frame += 1
        print(f"Processing video: {current_frame / total_frames * 100:.2f}% complete", end="\r")

    writer.release()
    vs.release()

def process_image(image_path, output_path):
    image = cv2.imread(image_path)
    image = detect_helmet(image, LABELS, COLORS, net, ln)
    cv2.imwrite(output_path, image)
    print("Helmet detection complete. Output saved at:", output_path)

# Example usage
FILE_NAME = "input_video.mp4"
process_video(f"data/{FILE_NAME}", f"data/{FILE_NAME.split('.')[0]}_output.{FILE_NAME.split('.')[1]}")
#process_image(f"data/{FILE_NAME}", f"data/{FILE_NAME.split('.')[0]}_output.{FILE_NAME.split('.')[1]}")