import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4
# output_path = "insert_path" 

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data --> Added batch Size
    model.fit(x_train, y_train, batch_size=32, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category. 
    Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """  

    image_extension = ".ppm"
    features = []
    labels = []

    for label_name in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, label_name)

        # Jump if is not a dir
        if not os.path.isdir(subdir_path):
            continue            
        try:
            label = int(label_name)
        except ValueError:
            continue # Jump if is not numerical value
                

        for file in os.listdir(subdir_path):
            if file.lower().endswith(image_extension):
                full_path = os.path.join(subdir_path, file)
                # save img as ndarray with 3 dim
                img = cv2.imread(full_path) 
                
                if img is not None:
                    # Resize and Normalize each pixel values to be between 0 and 1 --> / 255.0
                    # INTER_AREA --> best solution
                    resized_img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA).astype('float32') / 255.0               

                    # Verifying Object type
                    # print(f"Tipo: {type(resized_img)}") 
                    # Verifying shape (h, w, levenls)
                    # print(f"Dimensioni (Shape): {resized_img.shape}")


                    features.append(resized_img)
                    labels.append(label)
            
    tuple_collection = (features, labels)

    print("completed!")
    # Verifying Structure
    # print(f"Formato tuple: {type(dati_finali)}")
    # print(len(dati_finali))


    return tuple_collection[0], tuple_collection[1]
        

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    # Define Convolutional Neural Network
    model = tf.keras.models.Sequential([

        # First Convolution block, using 3x3 kernel
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_WIDTH,IMG_HEIGHT,3)),
        # Gaussian Normalizing for entire batch
        tf.keras.layers.BatchNormalization(),

        # Max-pooling layer, using 2x2 pool size
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # First Convolution block
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),

        # Terzo Blocco (Opzionale, più profondo)
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'), # No padding 
        
        # Flatten units
        tf.keras.layers.Flatten(),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        
        # Output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])
    model.summary()

    # Train neural network
    model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
    )

    # Return the model
    return model


if __name__ == "__main__":
    main()