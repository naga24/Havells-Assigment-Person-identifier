from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, classification_report
import numpy as np

# Load the trained model
model = load_model('model.h5')

# Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8

# Validation data generator
valid_datagen = ImageDataGenerator(rescale=1./255)

valid_generator = valid_datagen.flow_from_directory(
    './data/val/',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',  # Assuming binary classification
    shuffle=False
)

# Make predictions on the validation set
y_true = valid_generator.classes
num_samples = len(valid_generator)
y_pred_probs = model.predict(valid_generator, steps=num_samples)

# Convert probabilities to binary predictions (0 or 1)
y_pred_binary = (y_pred_probs > 0.5).astype(int)

# Calculate confusion matrix
conf_mat = classification_report(y_true, y_pred_binary)
print("Classification Report:")
print(conf_mat)

# Calculate F1 score
f1 = f1_score(y_true, y_pred_binary, average='binary')
print(f"F1 Score: {f1}")
