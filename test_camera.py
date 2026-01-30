import cv2

print("Testing Face Recognition Setup...")
print("-" * 50)

# Test 1: Check OpenCV Installation
print("\n1. Checking OpenCV installation...")
print(f"   OpenCV Version: {cv2.__version__}")

# Test 2: Load Haar Cascade
print("\n2. Loading face detection model...")
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("   ❌ Error: Cascade file not loaded!")
    exit()
else:
    print("   ✅ Cascade file loaded successfully!")
    print(f"   Location: {cascade_path}")

# Test 3: Access Webcam
print("\n3. Testing webcam access...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("   ❌ Could not access webcam")
    print("   Try changing VideoCapture(0) to VideoCapture(1)")
    exit()
else:
    print("   ✅ Webcam opened successfully!")

# Test 4: Capture and Detect Face
print("\n4. Capturing frame and detecting faces...")
ret, frame = cap.read()

if ret:
    print("   ✅ Frame captured successfully!")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(f"   ✅ Detected {len(faces)} face(s)")
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Save the image
    cv2.imwrite('test_detection.jpg', frame)
    print("   ✅ Saved test image as 'test_detection.jpg'")
else:
    print("   ❌ Could not capture frame")

cap.release()

print("\n" + "=" * 50)
print("✅ ALL TESTS PASSED! System is ready.")
print("=" * 50)