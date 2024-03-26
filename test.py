from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('model.h5')

# Load and preprocess the input image
img_path = './streamlit_app/demo_images/zomato_delivery_partner_Image_35.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.  # Normalize the image

# Make predictions
predictions = model.predict(img_array)

print(predictions)

# Print the predicted class and probability
if predictions[0] > 0.5:
    print("Predicted class: zomato")
else:
    print("Predicted class: other")

print("Predicted probability:", predictions[0])
