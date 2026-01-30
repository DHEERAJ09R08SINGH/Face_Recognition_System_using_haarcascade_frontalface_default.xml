from face_recognition_system import FaceRecognitionSystem

def main():
    """Main workflow for Face Recognition Project"""
    
    print("\n" + "="*60)
    print("🎯 FACE RECOGNITION PROJECT")
    print("="*60)
    
    # Initialize system - CHANGED TO INDEX 0
    system = FaceRecognitionSystem(camera_index=0)
    
    while True:
        print("\n📋 MENU:")
        print("1. Collect face data for a new person")
        print("2. Train the model")
        print("3. Test real-time recognition")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Collect data
            name = input("\nEnter person's name: ").strip()
            
            # Remove quotes and invalid characters
            name = name.strip('"').strip("'").strip()
            
            # Remove Windows invalid characters
            invalid_chars = '<>:"/\\|?*'
            for char in invalid_chars:
                name = name.replace(char, '')
            
            if name:
                try:
                    num_input = input("Number of images to collect (default 150): ").strip()
                    num_images = int(num_input) if num_input else 150
                    
                    if num_images < 50:
                        print("⚠️  Warning: Less than 50 images may result in poor accuracy")
                        confirm = input("Continue anyway? (y/n): ").strip().lower()
                        if confirm != 'y':
                            continue
                    
                    print(f"\n✅ Collecting data for: {name}")
                    system.collect_face_data(name, num_images)
                    
                except ValueError:
                    print("❌ Invalid number! Using default (150)")
                    system.collect_face_data(name, 150)
                except Exception as e:
                    print(f"❌ Error: {e}")
            else:
                print("❌ Invalid name!")
        
        elif choice == '2':
            # Train model
            try:
                system.train_model_simple()
            except Exception as e:
                print(f"❌ Error during training: {e}")
        
        elif choice == '3':
            # Real-time recognition
            try:
                if system.load_model_simple():
                    system.recognize_faces_live()
            except Exception as e:
                print(f"❌ Error during recognition: {e}")
        
        elif choice == '4':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice! Please enter 1, 2, 3, or 4")

if __name__ == "__main__":
    main()