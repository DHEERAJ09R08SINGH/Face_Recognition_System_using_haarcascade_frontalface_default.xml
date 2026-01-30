import os

# Create project directories
directories = [
    'dataset',           # Will store face images
    'models',           # Will store trained models
    'plots',            # Will store training graphs
    'test_images'       # For testing
]

print("Creating project structure...")
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"✅ Created: {directory}/")

print("\n✅ Project structure ready!")
print("\nYour project folder now has:")
print("├── dataset/          # Training data will go here")
print("├── models/           # Trained models will be saved here")
print("├── plots/            # Training graphs")
print("└── test_images/      # Test photos")