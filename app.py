"""
Face Recognition System using Deep Learning
Author: Dheeraj R Singh
Organization: Geekslab Technologies Pvt. Ltd.
Duration: Oct 2021 - Nov 2021

This system uses CNNs and transfer learning to identify individuals based on facial features.
Includes data scraping, preprocessing, model training, and real-time recognition.
"""

import cv2
import numpy as np
import os
import pickle
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class FaceRecognitionSystem:
    """Complete Face Recognition System with training and inference capabilities"""
    
    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size
        self.model = None
        self.label_encoder = LabelEncoder()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def collect_face_data(self, person_name, num_samples=100, save_dir='dataset'):
        """
        Collect face data using webcam for training
        
        Args:
            person_name: Name of the person
            num_samples: Number of face images to collect
            save_dir: Directory to save collected images
        """
        person_dir = os.path.join(save_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        count = 0
        
        print(f"Collecting {num_samples} samples for {person_name}...")
        print("Press 'q' to quit early")
        
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                count += 1
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, self.img_size)
                
                cv2.imwrite(
                    os.path.join(person_dir, f'{person_name}_{count}.jpg'),
                    face_resized
                )
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f'Collected: {count}/{num_samples}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            cv2.imshow('Collecting Faces - Press Q to quit', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q') or count >= num_samples:
                break
                
        cap.release()
        cv2.destroyAllWindows()
        print(f"Collected {count} samples for {person_name}")
        
    def load_dataset(self, dataset_dir='dataset'):
        """
        Load and preprocess dataset from directory
        
        Args:
            dataset_dir: Directory containing face images organized by person
            
        Returns:
            X: Preprocessed images
            y: Encoded labels
        """
        X, y = [], []
        
        for person_name in os.listdir(dataset_dir):
            person_path = os.path.join(dataset_dir, person_name)
            if not os.path.isdir(person_path):
                continue
                
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    img_resized = cv2.resize(img, self.img_size)
                    img_normalized = img_resized / 255.0
                    X.append(img_normalized)
                    y.append(person_name)
        
        X = np.array(X).reshape(-1, self.img_size[0], self.img_size[1], 1)
        y = self.label_encoder.fit_transform(y)
        
        return X, y
    
    def build_cnn_model(self, num_classes):
        """
        Build custom CNN architecture for face recognition
        
        Args:
            num_classes: Number of people to recognize
            
        Returns:
            Compiled CNN model
        """
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size[0], self.img_size[1], 1)),
            MaxPooling2D((2, 2)),
            
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_transfer_learning_model(self, num_classes):
        """
        Build model using VGG16 transfer learning
        
        Args:
            num_classes: Number of people to recognize
            
        Returns:
            Compiled transfer learning model
        """
        # Load pre-trained VGG16 (without top layers)
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size[0], self.img_size[1], 3)
        )
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Add custom classification layers
        x = Flatten()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X, y, model_type='cnn', epochs=50, batch_size=32):
        """
        Train the face recognition model
        
        Args:
            X: Training images
            y: Training labels
            model_type: 'cnn' or 'transfer_learning'
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        num_classes = len(np.unique(y))
        
        # Build model
        if model_type == 'transfer_learning':
            # Convert grayscale to RGB for VGG16
            X_train_rgb = np.repeat(X_train, 3, axis=-1)
            X_val_rgb = np.repeat(X_val, 3, axis=-1)
            self.model = self.build_transfer_learning_model(num_classes)
            X_train, X_val = X_train_rgb, X_val_rgb
        else:
            self.model = self.build_cnn_model(num_classes)
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )
        
        # Callbacks
        checkpoint = ModelCheckpoint(
            'best_face_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=[checkpoint, early_stop],
            verbose=1
        )
        
        self.plot_training_history(history)
        
        return history
    
    def plot_training_history(self, history):
        """Plot training and validation accuracy/loss"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Val Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        
    def save_model(self, model_path='face_recognition_model.h5', 
                   encoder_path='label_encoder.pkl'):
        """Save trained model and label encoder"""
        self.model.save(model_path)
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"Model saved to {model_path}")
        print(f"Label encoder saved to {encoder_path}")
    
    def load_trained_model(self, model_path='face_recognition_model.h5',
                          encoder_path='label_encoder.pkl'):
        """Load trained model and label encoder"""
        self.model = load_model(model_path)
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print("Model loaded successfully")
    
    def recognize_faces_realtime(self):
        """Real-time face recognition using webcam"""
        if self.model is None:
            print("Error: No model loaded. Train or load a model first.")
            return
        
        cap = cv2.VideoCapture(0)
        print("Starting real-time face recognition. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, self.img_size)
                face_normalized = face_resized / 255.0
                face_input = face_normalized.reshape(1, self.img_size[0], self.img_size[1], 1)
                
                # Predict
                predictions = self.model.predict(face_input, verbose=0)
                confidence = np.max(predictions)
                predicted_class = np.argmax(predictions)
                person_name = self.label_encoder.inverse_transform([predicted_class])[0]
                
                # Draw rectangle and label
                color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                label = f'{person_name}: {confidence*100:.1f}%'
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow('Face Recognition - Press Q to quit', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


# Example Usage
if __name__ == "__main__":
    # Initialize system
    fr_system = FaceRecognitionSystem(img_size=(128, 128))
    
    # Step 1: Collect training data (run this for each person)
    # Uncomment to collect data:
    # fr_system.collect_face_data('Person1', num_samples=100)
    # fr_system.collect_face_data('Person2', num_samples=100)
    # fr_system.collect_face_data('Person3', num_samples=100)
    
    # Step 2: Load dataset
    print("Loading dataset...")
    X, y = fr_system.load_dataset('dataset')
    print(f"Loaded {len(X)} images with {len(np.unique(y))} classes")
    
    # Step 3: Train model
    print("\nTraining model...")
    history = fr_system.train_model(X, y, model_type='cnn', epochs=50)
    
    # Step 4: Save model
    fr_system.save_model()
    
    # Step 5: Real-time recognition
    print("\nStarting real-time face recognition...")
    fr_system.recognize_faces_realtime()
    
    # Alternative: Load existing model and run recognition
    # fr_system.load_trained_model()
    # fr_system.recognize_faces_realtime()