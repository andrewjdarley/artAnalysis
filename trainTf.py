# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import os
from matplotlib import pyplot as plt

# Define the path to your dataset
data_dir = '/mnt/c/Users/andre/OneDrive/Desktop/Research/caravaggio/snippets'

# Filter subdirectories that have at least 150 images
valid_subdirs = [os.path.join(data_dir, subdir) for subdir in os.listdir(data_dir) 
                 if os.path.isdir(os.path.join(data_dir, subdir)) and 
                 len(os.listdir(os.path.join(data_dir, subdir))) >= 250]

if not valid_subdirs:
    raise ValueError("No subdirectories with at least 150 images found.")

# Initialize the ImageDataGenerator for train and validation sets
datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)  # 80-20 train-validation split

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training',  # set as training data
    classes=[os.path.basename(subdir) for subdir in valid_subdirs]  # only include valid subdirs
)

validation_generator = datagen.flow_from_directory(
    data_dir,   
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',  # set as validation data
    classes=[os.path.basename(subdir) for subdir in valid_subdirs]  # only include valid subdirs
)

# Get the number of classes
num_classes = len(train_generator.class_indices)
print('CLASSES:', num_classes)

# Load the ResNet50 model without the top classification layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze all layers in ResNet50
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with a specified learning rate
initial_learning_rate = 0.000001
model.compile(optimizer=Adam(learning_rate=initial_learning_rate), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50
)

# Plot accuracy and loss for training and validation data
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.savefig('accuracy_loss_plot.png')
plt.show()

# Save the model
model.save('model.weights.h5')
