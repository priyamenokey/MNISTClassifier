# MNISTClassifier
A neural network training pipeline (with Keras framework) for MNIST handwritten digits classification task.

A dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images is loaded from Keras.Model is trained through Convnets with dropout regularization also performed .Model gives an accuracy of 99.9 % and is saved .In testMNIST.py model is loaded and image probability is infered .The treshold value for prediction is 60%.Command for executing inference is given below 


python3 testMNIST.py --model keras_mnist.h5 --image imagetest/img_235.jpg

