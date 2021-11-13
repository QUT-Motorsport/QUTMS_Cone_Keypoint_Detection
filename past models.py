from keras import layers

model_in = layers.Input(shape=(100, 100, 3))
layer_out = layers.Conv2D(64, (7, 7), input_shape=(100, 100, 3))(model_in)
layer_out = layers.BatchNormalization()(layer_out)
layer_out = layers.ReLU()(layer_out)
# layer_out = layers.MaxPooling2D((2, 2))(layer_out)
# layer_out = layers.Conv2D(64, (3, 3), activation='relu')(layer_out)
# layer_out = layers.MaxPooling2D((2, 2))(layer_out)
# layer_out = layers.Conv2D(64, (3, 3), activation='relu')(layer_out)
layer_out = layers.Flatten()(layer_out)
# layer_out = layers.Dense(128, activation='relu')(layer_out)
layer_out = layers.Dense(64, activation='relu')(layer_out)
layer_out = layers.Dense(128, activation='relu')(layer_out)
layer_out = layers.Dense(256, activation='relu')(layer_out)
layer_out = layers.Dense(512, activation='relu')(layer_out)
model_out = layers.Dense(16, activation='relu')(layer_out)