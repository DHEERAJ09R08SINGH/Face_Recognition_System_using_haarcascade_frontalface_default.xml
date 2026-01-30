import cv2

print("Testing all possible camera configurations...\n")

backends = [
    ("Default", None),
    ("DirectShow", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
]

for idx in range(5):
    for backend_name, backend in backends:
        if backend:
            cap = cv2.VideoCapture(idx, backend)
        else:
            cap = cv2.VideoCapture(idx)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera {idx} with {backend_name}: WORKS!")
                cap.release()
                break
        cap.release()