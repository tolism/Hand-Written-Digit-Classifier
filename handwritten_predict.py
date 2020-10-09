'''
 FILE: handwritten_predict.py
 Tester for the CNN MNIST Predictor
 Author:
   Moustaklis Apostolos , amoustakl@auth.gr
'''
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load and prepare the image


def load_image(filename):
    # Load the image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    # Convert to array
    img = img_to_array(img)
    # Reshape the image to grayscale
    img = img.reshape(1, 28, 28, 1)
    # Prepare pixel data from [0-255] -> [0-1.00]
    img = img.astype('float32')
    img = img / 255.0
    return img

# Load an image and predict the class


def run_example():
    path = 'number_samples/'
    filename = 'sample_image_Z.png'

    for i in range(0, 10):
        filename = filename[:13]+str(i)+filename[13+1:]
        # Load the image
        img = load_image(path+filename)
        # Load model
        model = load_model('final_model.h5')
        # Predict the class
        digit = model.predict_classes(img)
        print(filename)
        print(digit[0])

# Run the example


run_example()
