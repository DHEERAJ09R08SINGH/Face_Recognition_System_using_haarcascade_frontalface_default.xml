import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import time

class FaceRecognitionSystem:
    def __init__(self, camera_index=1):
        """
        Initialize Face Recognition System
        
        Args:
            camera_index: Camera device index (1 for DroidCam, 0 for built-in)
        """
        # Setup directories
        self.data_dir = 'dataset'
        self.model_dir = 'models'
        self.plot_dir = 'plots'
        
        for directory in [self.data_dir, self.model_dir, self.plot_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Load face detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Camera settings
        self.camera_index = camera_index
        self.img_size = (128, 128)
        
        # Model will be loaded later
        self.model = None
        self.label_encoder = None
        
        print("✅ Face Recognition System initialized!")
    
    def collect_face_data(self, person_name, num_images=150):
        """
        Collect face images for training
        
        Args:
            person_name: Name of the person
            num_images: Number of images to collect (default: 150)
        """
        person_dir = os.path.join(self.data_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print(f"\n{'='*60}")
        print(f"📸 Collecting images for: {person_name}")
        print(f"{'='*60}")
        print("Instructions:")
        print("- Look at the camera")
        print("- Move your head slightly (left, right, up, down)")
        print("- Try different expressions (smile, neutral, serious)")
        print("- Press 'q' to stop early")
        print(f"Target: {num_images} images")
        print(f"{'='*60}\n")
        
        # Give camera time to warm up
        time.sleep(2)
        
        count = 0
        
        while count < num_images:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # FIXED: Added self. prefix
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            
            # Display frame
            display_frame = frame.copy()
            
            for (x, y, w, h) in faces:
                # Draw rectangle
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Extract and save face
                face_roi = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, self.img_size)
                
                # Save image
                img_path = os.path.join(person_dir, f'face_{count}.jpg')
                cv2.imwrite(img_path, face_resized)
                count += 1
                
                # Show progress
                cv2.putText(display_frame, f"Captured: {count}/{num_images}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # If no face detected
            if len(faces) == 0:
                cv2.putText(display_frame, "No face detected - Please face camera", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow(f'Collecting Data for {person_name}', display_frame)
            
            # Press 'q' to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠️  Stopped by user")
                break
            
            # Small delay between captures for variety
            time.sleep(0.1)
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✅ Collected {count} images for {person_name}")
        print(f"📁 Saved in: {person_dir}/\n")

    def load_dataset(self):
        """Load all images from dataset directory"""
        print("\n📂 Loading dataset...")
        
        images = []
        labels = []
        
        # Get all person directories
        if not os.path.exists(self.data_dir):
            print("❌ Dataset directory not found!")
            return None, None
        
        person_dirs = [d for d in os.listdir(self.data_dir) 
                      if os.path.isdir(os.path.join(self.data_dir, d))]
        
        if len(person_dirs) == 0:
            print("❌ No person folders found in dataset!")
            return None, None
        
        print(f"Found {len(person_dirs)} people: {', '.join(person_dirs)}")
        
        for person_name in person_dirs:
            person_path = os.path.join(self.data_dir, person_name)
            image_files = [f for f in os.listdir(person_path) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"  - {person_name}: {len(image_files)} images")
            
            for img_file in image_files:
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path)
                
                if img is not None:
                    img = cv2.resize(img, self.img_size)
                    images.append(img)
                    labels.append(person_name)
        
        print(f"\n✅ Loaded {len(images)} total images")
        
        return np.array(images), np.array(labels)
    
    def train_model_simple(self, test_size=0.2):
        """
        Train a simple face recognition model using OpenCV's LBPH
        (Faster and simpler - good for quick testing)
        """
        print("\n" + "="*60)
        print("🎓 TRAINING SIMPLE MODEL (LBPH)")
        print("="*60)
        
        # Load dataset
        images, labels = self.load_dataset()
        
        if images is None:
            return
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Convert to grayscale for LBPH
        gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            gray_images, labels_encoded, test_size=test_size, random_state=42
        )
        
        print(f"\nTraining set: {len(X_train)} images")
        print(f"Test set: {len(X_test)} images")
        
        # Train LBPH model
        print("\n🔄 Training model...")
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.model.train(X_train, np.array(y_train))
        
        # Evaluate
        print("\n📊 Evaluating model...")
        correct = 0
        for img, true_label in zip(X_test, y_test):
            label, confidence = self.model.predict(img)
            if label == true_label:
                correct += 1
        
        accuracy = (correct / len(X_test)) * 100
        print(f"✅ Test Accuracy: {accuracy:.2f}%")
        
        # Save model
        model_path = os.path.join(self.model_dir, 'lbph_model.yml')
        encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
        
        self.model.write(model_path)
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"\n💾 Model saved to: {model_path}")
        print(f"💾 Encoder saved to: {encoder_path}")
        print("="*60)
    
    def load_model_simple(self):
        """Load the trained LBPH model"""
        model_path = os.path.join(self.model_dir, 'lbph_model.yml')
        encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
        
        if not os.path.exists(model_path):
            print("❌ Model not found! Please train first.")
            return False
        
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.model.read(model_path)
        
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print("✅ Model loaded successfully!")
        return True
    
    def recognize_faces_live(self):
        """Real-time face recognition"""
        if self.model is None:
            print("❌ No model loaded! Please train or load a model first.")
            return
        
        print("\n" + "="*60)
        print("🎥 REAL-TIME FACE RECOGNITION")
        print("="*60)
        print("Press 'q' to quit")
        print("="*60 + "\n")
        
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            
            for (x, y, w, h) in faces:
                # Extract face
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, self.img_size)
                
                # Predict
                label, confidence = self.model.predict(face_resized)
                person_name = self.label_encoder.inverse_transform([label])[0]
                
                # Confidence threshold (lower is better for LBPH)
                if confidence < 100:  # Adjust this threshold
                    text = f"{person_name} ({confidence:.0f})"
                    color = (0, 255, 0)  # Green
                else:
                    text = f"Unknown ({confidence:.0f})"
                    color = (0, 0, 255)  # Red
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Recognition stopped")


# For easy importing
if __name__ == "__main__":
    print("Face Recognition System Module")
    print("Import this in your main script")