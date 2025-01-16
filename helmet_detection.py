"""
AUTHOR: Rishi Bagul
USAGE: python Helmet-detection.py


Python env - /Users/user/Documents/Project/1_Face\ Detection/.venv/bin/python #- this is for me
""" 



# Import the necessary packages
import numpy as np
import time
import cv2
import os
import pyttsx3

# Hardcoded inputs for YOLO directory, input video, and output video
YOLO_DIR = "yolo-coco"  # Replace with the path to your YOLO directory
INPUT_VIDEO = "data/input_video.mp4"  # Replace with your input video file path
OUTPUT_VIDEO = "data/output_video.mp4"  # Replace with your desired output video path
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load the COCO class labels this YOLO model was trained on
labelsPath = os.path.sep.join([YOLO_DIR, "cocohelmet.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# Initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([YOLO_DIR, "yolov3-obj_2400.weights"])
configPath = os.path.sep.join([YOLO_DIR, "yolov3-obj.cfg"])

# Load the YOLO object detector trained on COCO dataset
print("Ready to Load")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Get the output layer names from YOLO
layer_names = net.getLayerNames()
try:
    ln = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except AttributeError:  # Handle older OpenCV versions
    ln = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize video stream and pointer to output video file
vs = cv2.VideoCapture(INPUT_VIDEO)  # Replace with your video file path

writer = None
(W, H) = (None, None)

# Get video properties (frame rate)
fps = vs.get(cv2.CAP_PROP_FPS)
# Retrieve the total number of frames in the video
total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize frame counter
current_frame = 0
# Loop over frames from the video file stream
while True:
    # Read the next frame from the file
    ret, frame = vs.read()
    if not ret:
        print("End of video file or error reading the file")
        break
    #print("In frame")
    current_frame += 1
    # If the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # Construct a blob from the input frame and perform a forward pass of YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # Initialize lists for detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    classIDs = []

    # Loop over each of the layer outputs
    for output in layerOutputs:
        for detection in output:
            # Extract the class ID and confidence
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Filter out weak predictions
            if confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Update lists
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply non-maxima suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # Ensure at least one detection exists
    if len(idxs) > 0:
        # Loop over the indexes we are keeping
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Draw a bounding box rectangle and label
            color = [int(c) for c in COLORS[classIDs[i]]] 
            #color = (0, 0, 255) # for red
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 3)
            textname = LABELS[classIDs[i]]
            #print("I just found", textname)
            a = str(textname)
            #engine.say("I found " + a) #if helmet is detected or not detected
        engine.runAndWait()

    # Initialize video writer if needed
    if writer is None:
        # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame.shape[1], frame.shape[0]), True)
    # Write the output frame to disk
    writer.write(frame)
    # Calculate progress and display it
    progress = (current_frame / total_frames) * 100
    print(f"Processing video: {progress:.2f}% complete", end="\r")
# Release the file pointers
print("Cleaning up the stuff...")
writer.release()
vs.release()
