# import the necessary packages
from pyimagesearch.simplecnn import SimpleCNN
from pyimagesearch.datagen import generate_adversarial_batch
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
import numpy as np

# load CIFAR-10 dataset and scale the pixel values to the range [0, 1]
print("[INFO] loading CIFAR-10 dataset...")
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0
# one-hot encode our labels
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

# initialize our optimizer and model
print("[INFO] compiling model...")
opt = Adam(0.001)
model = SimpleCNN.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# train the simple CNN on CIFAR-10
print("[INFO] training network...")
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=5, verbose=1)

# make predictions on the testing set for the model trained on
# non-adversarial images
(loss, acc) = model.evaluate(x=testX, y=testY, verbose=0)
print("[INFO] normal testing images:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))

# generate a set of adversarial examples with FGSM
print("[INFO] generating adversarial examples with FGSM...\n")
(advX, advY) = next(generate_adversarial_batch(model, len(testX), testX, testY, (32, 32, 3), eps=0.1))

# re-evaluate the model on the adversarial images
(loss, acc) = model.evaluate(x=advX, y=advY, verbose=0)
print("[INFO] adversarial testing images:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))

# lower the learning rate and re-compile the model for fine-tuning
print("[INFO] re-compiling model...")
opt = Adam(0.0001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# fine-tune the model on the adversarial examples
print("[INFO] fine-tuning network on adversarial examples...")
model.fit(advX, advY, batch_size=64, epochs=5, verbose=1)

# evaluate the model on normal testing images again after fine-tuning
(loss, acc) = model.evaluate(x=testX, y=testY, verbose=0)
print("[INFO] normal testing images *after* fine-tuning:")
print("[INFO] loss: {:.4f}, acc: {:.4f}\n".format(loss, acc))

# evaluate the model on adversarial images again after fine-tuning
(loss, acc) = model.evaluate(x=advX, y=advY, verbose=0)
print("[INFO] adversarial images *after* fine-tuning:")
print("[INFO] loss: {:.4f}, acc: {:.4f}".format(loss, acc))
