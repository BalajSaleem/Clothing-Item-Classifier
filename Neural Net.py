import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras

data = keras.datasets.fashion_mnist
(train_images,train_labels) , (test_images,test_labels) = data.load_data()

class_names = ['T-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
#condensing data to make it more managable
train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
    #the pictures are grids of 28x28 values(representing gray scale) so we flatten it to 1D array as the input later
    keras.layers.Flatten(input_shape=(28,28)),
    #make a dense(fully connected) layered nueral net with 128 neurons and use the activation function as relu
    keras.layers.Dense(128, activation="relu"),
    #another dense(fully connected) output layer figuring probabilities for each class.
    keras.layers.Dense(10, activation="softmax")
])
#setting up parameters for the model study optimizer and loss and our main factor is accuracy
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#epochs are repetitions
model.fit(train_images, train_labels, epochs=5)
#use test data to get the accuracy and loss
test_loss, test_acc = model.evaluate(test_images, test_labels)

prediction = model.predict(test_images)


for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()
#print(class_names[np.argmax(prediction[0])])

