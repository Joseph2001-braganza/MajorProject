import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load the ResNet50 model pretrained on ImageNet data
base_model = ResNet50(weights='imagenet', include_top=False)

# Prepare your dataset (e.g., CIFAR-10)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Add custom layers on top of the ResNet50 base model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
learning_rate = 1e-3
opt = Adam(learning_rate)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=64, epochs=20, verbose=1)

# Save model weights
model.save_weights('model_weights_resnet50.h5')