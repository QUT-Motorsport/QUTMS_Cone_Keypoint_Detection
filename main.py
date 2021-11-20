from matplotlib import pyplot as plt

from keras import layers, models

from get_data import get_data
from helper import *
####################### DATA #######################
train_images, val_images, test_images, train_labels, val_labels, test_labels = get_data(train_split=1, val_split=0)

print("train size:", np.array(train_labels).shape)
print("val size:", np.array(val_labels).shape)
print("test size:", np.array(test_labels).shape)

####################### MODEL #######################
# input
model_in = layers.Input(shape=(100, 100, 3))

# conv layer
layer_out = layers.Conv2D(64, (7, 7))(model_in)
layer_out = layers.ReLU()(layer_out)
layer_out = layers.BatchNormalization()(layer_out)


# Resnet layer
resnet_layers = [128, 128, 256]#, 512]
for layer_size in resnet_layers:
    print(layer_size)

    layer_out = layers.Conv2D(layer_size, (1, 1), padding='same')(layer_out)
    layer_out = layers.ReLU()(layer_out)
    layer_out = layers.BatchNormalization()(layer_out)

    skip_input = layer_out

    layer_out = layers.Conv2D(layer_size, (3, 3), padding='same')(layer_out)
    layer_out = layers.ReLU()(layer_out)
    layer_out = layers.BatchNormalization()(layer_out)

    layer_out = layers.Conv2D(layer_size, (3, 3), padding='same')(layer_out)
    layer_out = layers.ReLU()(layer_out)
    layer_out = layers.BatchNormalization()(layer_out)

    layer_out = layers.Add()([layer_out, skip_input])
    layer_out = layers.ReLU()(layer_out)

    # layer_out = layers.Conv2D(layer_size, (3, 3))(layer_out)
    # # I think this is wrong and res_input should actually be set before the first conv2d but
    # # that causes dimension errors so...
    # res_input = layer_out
    # layer_out = layers.BatchNormalization()(layer_out)
    # layer_out = layers.ReLU()(layer_out)
    # layer_out = layers.Conv2D(layer_size, (3, 3), padding='same')(layer_out)
    # layer_out = layers.BatchNormalization()(layer_out)
    # layer_out = layers.Add()([layer_out, res_input])
    # layer_out = layers.ReLU()(layer_out)

# output
layer_out = layers.Flatten()(layer_out)
# layer_out = layers.ReLU()(layer_out)
model_out = layers.Dense(32)(layer_out)
model_out = layers.Dense(16)(layer_out)

model = models.Model(inputs=model_in, outputs=model_out)
model.compile(optimizer='rmsprop', loss=is_this_loss, run_eagerly=True)
model.summary()

####################### TRAINING #######################

checkpoint_path = "model/saved"
try:
    model.load_weights(checkpoint_path)
    print("successfully loaded model")
except Exception as e:
    print(e)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1,
    monitor='loss',
    mode='min',
    save_best_only=True)

history = model.fit(train_images, train_labels, epochs=1000000, batch_size=64, callbacks=[model_checkpoint_callback])


####################### VISUALISATION #######################
# model - 21-11-13.save_weights(checkpoint_path)
print(len(train_images))
ims = train_images[-30:]
labs = train_labels[-30:]
pred = model.predict(ims)

# print(train_images)
# print("pred")
# print(pred)
# print("cool")

for i, image in enumerate(ims):
    plt.imshow(ims[i])
    plt.scatter((labs[i].reshape(8, 2).transpose()[0] + 0) * 1, (labs[i].reshape(8, 2).transpose()[1] + 0) * 1)
    plt.scatter((pred[i].reshape(8, 2).transpose()[0] + 0) * 1, (pred[i].reshape(8, 2).transpose()[1] + 0) * 1)
    plt.plot((labs[i].reshape(8, 2).transpose()[0] + 0) * 1, (labs[i].reshape(8, 2).transpose()[1] + 0) * 1)
    plt.plot((pred[i].reshape(8, 2).transpose()[0] + 0) * 1, (pred[i].reshape(8, 2).transpose()[1] + 0) * 1)

    print("#" * 30)
    print(labs[i].reshape(8, 2))
    print(pred[i].reshape(8, 2))
    print(labs[i].reshape(8, 2) - pred[i].reshape(8, 2))
    print(np.mean(np.square(labs[i].reshape(8, 2) - pred[i].reshape(8, 2))))

    plt.show()

