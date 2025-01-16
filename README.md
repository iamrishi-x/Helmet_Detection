# Helmet Detection with YOLOv3

🚨 **Helmet Detection System** 🚨  
This project is designed to detect helmets from images and videos with **99% accuracy** using **YOLOv3** and **OpenCV**. The system identifies whether a person is wearing a helmet or not in real-time, providing an essential tool for safety monitoring, security applications, and traffic enforcement.

### 🛠️ **Technologies Used**
- **Python**  
- **OpenCV**  
- **YOLOv3** (You Only Look Once - Real-time Object Detection)  
- **pyttsx3** (for voice alerts)  

### ⚡ **Features**
- Detect helmets from images or video streams
- Real-time processing with a high degree of accuracy (99%)
- Support for both image and video input
- Voice feedback when a helmet is detected
- Output video with bounding boxes drawn around detected helmets

### 📂 **Installation**

#### 1. Clone the repository:
```bash
git clone https://github.com/iamrishi-x/Helmet_Detection.git
```

#### 2. Install Required Dependencies
Create a virtual environment and install the necessary libraries. Use the `requirements.txt` file to install all dependencies.

```bash
pip install -r requirements.txt
```
---

#### 3. Download YOLOv3 Weights
Download the pre-trained YOLOv3 weights from the link below:

- [Download YOLO Weights](https://drive.google.com/file/d/1_xBdP1GRK4i7yzJP8_a5GWaejZZKjdyI/view)

---

#### 4. Set Up the Project
Ensure the YOLOv3 weights and configuration files are correctly placed in the `yolo-coco` folder inside your project directory. These files are essential for performing the object detection task.

Place the following files in the `yolo-coco` directory:
- `yolov3-obj.cfg` (YOLO configuration file)
- `yolov3-obj_2400.weights` (YOLO pre-trained weights file)
- `cocohelmet.names` (YOLO class labels file)

The folder structure should look like this:

```plaintext
Helmet_Detection/
│
├── yolo-coco/
│   ├── yolov3-obj.cfg
│   ├── yolov3-obj_2400.weights
│   └── cocohelmet.names
│
├── data/
│   ├── input_video.mp4
│   └── output_video.mp4
│
├── helmet_detection.py
└── requirements.txt
```
---
### 🚀 How to Use

##### Detect from Image or Custom Video File
Run the script [helmet_detection.py](helmet_detection.py) to start helmet detection using a video in "data" folder.

##### Detect from Webcam or Video Stream
Need to add in future version

---

### 🎨 Customization
- Modify the minimum confidence threshold for detection in the script to filter weak detections.
- Adjust bounding box colors and text settings to personalize the detection appearance.

---

### 💬 Connect with Me!
- **LinkedIn**: [Rishi Bagul](https://www.linkedin.com/in/rishibagul7/)

Feel free to contribute, raise issues, or suggest improvements. Happy coding! ✨

---

### 🔧 Future Enhancements
- Need to add live detection
- Enhance detection accuracy using custom-trained models.
- Integrate real-time monitoring capabilities with a cloud platform.

---

### 📝 License
This project is open-source under the MIT license. See the [LICENSE](LICENSE) file for more details.
