# Face Recognition System using Haar Cascade

A **real-time Face Recognition System** built using **Python and OpenCV**, utilizing the `haarcascade_frontalface_default.xml` classifier for fast and efficient face detection. The system detects human faces from a live webcam feed or images and is designed to be easily extended for recognition-based applications such as attendance systems and security solutions.

---

## 🚀 Features

* Real-time face detection using Haar Cascade Classifier
* Supports live webcam and static image input
* Fast, lightweight, and efficient processing
* Beginner-friendly and easy to understand codebase
* Scalable for face recognition and attendance systems

---

## 🛠️ Technologies Used

* Python
* OpenCV
* NumPy
* Haar Cascade Classifier

---

## 📁 Project Structure

```
face_recognition_project/
│
├── setup_project.py              # Creates required folders
├── face_recognition_system.py    # Core face detection & recognition logic
├── run_project.py                # Main entry point to run the project
├── test_camera.py                # Camera connectivity test
├── live_face_detection.py        # Live face detection script
│
├── dataset/                      # Training dataset
│   ├── John/
│   │   ├── face_0.jpg
│   │   ├── face_1.jpg
│   │   └── ... (up to 150 images)
│   │
│   └── Sarah/
│       └── ... (up to 150 images)
│
└── models/                       # Saved trained models
    ├── lbph_model.yml
    └── label_encoder.pkl
```

---

## 🧠 How It Works

* Captures frames from webcam or loads images
* Converts frames to grayscale for faster processing
* Detects faces using Haar Cascade Classifier
* Draws bounding boxes around detected faces
* Displays real-time detection output

---

## 📌 Applications

* Face-based attendance systems
* Security and surveillance
* Access control systems
* Human–Computer Interaction (HCI)

---

## 🔮 Future Enhancements

* Improve recognition accuracy using deep learning (CNN)
* Add automatic attendance logging
* Integrate database storage
* Deploy as a web application

---

## 👨‍💻 Author

**Dheeraj R. Singh**

📧 Email: [newagecoder09@gmail.com](mailto:newagecoder09@gmail.com)

🔗 GitHub: [https://github.com/DHEERAJ09R08SINGH](https://github.com/DHEERAJ09R08SINGH)

---
