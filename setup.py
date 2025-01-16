from setuptools import setup, find_packages

setup(
    name="helmet-detection",
    version="1.0.0",
    author="Rishi D. Bagul",
    author_email="rushibagul4444@gmail.com",
    description="A YOLO-based helmet detection system.",
    url="https://github.com/iamrishi-x/Helmet_Detection",
    packages=find_packages(),
    install_requires=[
        "numpy==1.26.4",
        "imutils==0.5.4",
        "opencv-python==4.10.0.84",
        "pyttsx3==2.98",
        "opencv-contrib-python==4.10.0.84"
    ],
)
