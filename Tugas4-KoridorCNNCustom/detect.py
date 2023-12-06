import cv2
import numpy as np
import tensorflow as tf
import keras.utils as image
import matplotlib.pyplot as plt

# Load your trained model
model = tf.keras.models.load_model('./model/custom_model.h5')

# Set the input size based on your trained model
input_size = (224, 224)

# Create a dictionary for class labels
class_labels = {0: 'koridor', 1: 'lift', 2: 'pintu_darurat', 3: 'pintu_ruangan', 4: 'tangga'}

# Function to perform real-time detection
def real_time_detection():
    cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera

    while True:
        ret, frame = cap.read()

        # Resize the frame to the input size expected by your model
        resized_frame = cv2.resize(frame, input_size)

        # Preprocess the frame
        img_array = tf.keras.preprocessing.image.img_to_array(resized_frame)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image

        # Perform prediction
        predictions = model.predict(img_array)
        score = predictions[0]
        predicted_class = np.argmax(score)
        predicted_label = class_labels[predicted_class]

        # Display the frame with the predicted class
        cv2.putText(frame, f'Prediction: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Real-time Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_detection()
