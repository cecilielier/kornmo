# Databricks notebook source
from sentinel.storage import SentinelDataset

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from matplotlib import pyplot as plt
import geopandas as gpd


CLASSES = ['bygg', 'rug', 'rughvete', 'hvete', 'havre']

# COMMAND ----------

labels_lookup = gpd.read_file('../../kornmo-data-files/raw-data/crop-classification-data/all_data.gpkg')
labels_lookup["key"] = labels_lookup.orgnr.astype(int).astype(str) + labels_lookup.index.astype(str) + "/" + labels_lookup.year.astype(int).astype(str)
labels_lookup.drop(columns=["year", "orgnr"], inplace=True)

labels_lookup.drop(labels_lookup[labels_lookup['area'] < 1500].index, inplace = True)
labels_lookup = labels_lookup.loc[labels_lookup['planted'] != 'erter']
labels_lookup = labels_lookup.loc[labels_lookup['planted'] != 'oljefro']

labels_lookup

# COMMAND ----------

labels_to_keep = [f"{int(label.split('/')[1])}/{int(label.split('/')[2])}" for label in sd.labels]

print(pd.Series(list(labels_lookup['planted'])).value_counts())
labels_lookup

# COMMAND ----------

sd = SentinelDataset('E:/MasterThesisData/Satellite_Images/small_images_all.h5')
all_images = sd.to_iterator().shuffled()

# COMMAND ----------

unique = labels_lookup['key'].unique()

data_x = []
data_y = []
max_imgs = 15000
i = 0

# Loads 15000 images including their labels directly into memory
for data in tqdm(all_images, total=max_imgs):

    data_x.append(data[2][()])
    if f"{int(data[0])}/{int(data[1])}" in unique:
        label = labels_lookup.loc[labels_lookup['key'] == f"{int(data[0])}/{int(data[1])}"]['planted'].iloc[0]
        data_y.append(label)
    else:
        data_y.append(None)
    if i >= max_imgs - 1:
        break
    i += 1


#print(np.array(data_x).shape)

# COMMAND ----------

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(data_x, data_y, test_size=0.3, random_state=42)
del data_x, data_y
print(f'x_train: {len(x_train)}')
print(f'x_val: {len(x_val)}')
print(f'y_train: {len(y_train)}')
print(f'y_val: {len(y_val)}')

# COMMAND ----------

print(x_train[0])
print(y_train[0])

# COMMAND ----------

BAG_COUNT = 100
VAL_BAG_COUNT = 100
BAG_SIZE = 7
PLOT_SIZE = 3

(BAG_COUNT + VAL_BAG_COUNT) * BAG_SIZE

# COMMAND ----------

def create_bags(input_data, input_labels, positive_class, bag_count, instance_count):
    bags = []
    bag_labels = []

    # input_data = np.divide(input_data, 255.0)

    count = 0
    for _ in tqdm(range(bag_count), desc=f'Creating bag for {positive_class}'):
        index = np.random.choice(input_data.shape[0], instance_count, replace=False)
        instances_data = input_data[index]
        instances_labels = input_labels[index]

        bag_label = 0

        if positive_class in instances_labels:
            bag_label = 1
            count += 1

        bags.append(instances_data)
        bag_labels.append(np.array([bag_label]))

    print(f"Positive bags: {count}")
    print(f"Negative bags: {bag_count - count}")
    return list(np.swapaxes(bags, 0, 1)), np.array(bag_labels)


train_data, train_labels = create_bags(np.array(x_train), np.array(y_train), 'rug', BAG_COUNT, BAG_SIZE)
val_data, val_labels = create_bags(np.array(x_val), np.array(y_val), 'rug', VAL_BAG_COUNT, BAG_SIZE)


# COMMAND ----------

class MILAttentionLayer(layers.Layer):

    def  __init__(
        self,
        weight_params_dim,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        use_gated=False,
        **kwargs
    ):

        super(MILAttentionLayer, self).__init__(**kwargs)

        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

        self.v_init = self.kernel_initializer
        self.w_init = self.kernel_initializer
        self.u_init = self.kernel_initializer

        self.v_regularizer = self.kernel_regularizer
        self.w_regularizer = self.kernel_regularizer
        self.u_regularizer = self.kernel_regularizer

    def get_config(self):
        config = super(MILAttentionLayer, self).get_config()
        config.update({"weight_params_dim": self.weight_params_dim})
        config.update({"use_gated": self.use_gated})
        config.update({"kernel_initializer": self.kernel_initializer})
        config.update({"kernel_regularizer": self.kernel_regularizer})

        return config

    def build(self, input_shape):

        # Input shape.
        # List of 2D tensors with shape: (batch_size, input_dim).
        input_dim = 48

        self.v_weight_params = self.add_weight(
            shape=(input_dim, self.weight_params_dim),
            initializer=self.v_init,
            name="v",
            regularizer=self.v_regularizer,
            trainable=True,
        )

        self.w_weight_params = self.add_weight(
            shape=(self.weight_params_dim, 1),
            initializer=self.w_init,
            name="w",
            regularizer=self.w_regularizer,
            trainable=True,
        )

        if self.use_gated:
            self.u_weight_params = self.add_weight(
                shape=(input_dim, self.weight_params_dim),
                initializer=self.u_init,
                name="u",
                regularizer=self.u_regularizer,
                trainable=True,
            )
        else:
            self.u_weight_params = None

        self.input_built = True

    def call(self, inputs):

        # Assigning variables from the number of inputs.
        instances = [self.compute_attention_scores(instance) for instance in inputs]

        # Apply softmax over instances such that the output summation is equal to 1.
        alpha = tf.math.softmax(instances, axis=0)

        return [alpha[i] for i in range(alpha.shape[0])]

    def compute_attention_scores(self, instance):

        # Reserve in-case "gated mechanism" used.
        original_instance = instance

        # tanh(v*h_k^T)
        instance = tf.math.tanh(tf.tensordot(instance, self.v_weight_params, axes=1))

        # for learning non-linear relations efficiently.
        if self.use_gated:

            instance = instance * tf.math.sigmoid(
                tf.tensordot(original_instance, self.u_weight_params, axes=1)
            )

        # w^T*(tanh(v*h_k^T)) / w^T*(tanh(v*h_k^T)*sigmoid(u*h_k^T))
        return tf.tensordot(instance, self.w_weight_params, axes=1)

# COMMAND ----------

def create_model(instance_shape):

    # Extract features from inputs.
    inputs, embeddings = [], []
    shared_dense_layer_1 = layers.Dense(96, activation="relu")
    dropout_layer = layers.Dropout(0.6)
    shared_dense_layer_2 = layers.Dense(48, activation="relu")
    for _ in range(BAG_SIZE):
        inp = layers.Input(instance_shape)
        flatten = layers.Flatten()(inp)
        dense_1 = shared_dense_layer_1(flatten)
        dropout = dropout_layer(dense_1)
        dense_2 = shared_dense_layer_2(dropout)
        inputs.append(inp)
        embeddings.append(dense_2)

    # Invoke the attention layer.
    alpha = MILAttentionLayer(weight_params_dim=256, kernel_regularizer=keras.regularizers.l2(0.001), use_gated=True, name="alpha")
    alpha = alpha(embeddings)

    # Multiply attention weights with the input layers.
    multiply_layers = [
        layers.multiply([alpha[i], embeddings[i]]) for i in range(len(alpha))
    ]

    # Concatenate layers.
    concat = layers.concatenate(multiply_layers, axis=1)

    # Classification output node.
    output = layers.Dense(2, activation="softmax")(concat)

    return keras.Model(inputs, output)

# COMMAND ----------

def compute_class_weights(labels):

    # Count number of postive and negative bags.
    negative_count = len(np.where(labels == 0)[0])
    positive_count = len(np.where(labels == 1)[0])
    total_count = negative_count + positive_count

    # Build class weight dictionary.
    return {
        0: (1 / negative_count) * (total_count / 2),
        1: (1 / positive_count) * (total_count / 2),
    }

# COMMAND ----------

from tensorflow import keras

def train(train_data, train_labels, val_data, val_labels, model):

    # Take the file name from the wrapper.
    file_path = "../src/grain_classification/training/mil/mil_model.h5"

    # Initialize model checkpoint callback.
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        file_path,
        monitor="val_loss",
        verbose=0,
        mode="min",
        save_best_only=True,
        save_weights_only=True,
    )

    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=100, mode="min", restore_best_weights=True)
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        train_data,
        train_labels,
        validation_data=(val_data, val_labels),
        epochs=150,
        class_weight=compute_class_weights(train_labels),
        batch_size=1,
        callbacks=[model_checkpoint, early_stopping],
        verbose=1,
    )

    # model.load_weights(file_path)

    return model


model = create_model((30, 16, 16, 12))
#print(model.summary())

trained_model = train(train_data, train_labels, val_data, val_labels, model)

# COMMAND ----------

import matplotlib.pyplot as plt

acc=trained_model.history.history['accuracy']
val_acc=trained_model.history.history['val_accuracy']
loss=trained_model.history.history['loss']
val_loss=trained_model.history.history['val_loss']

plt.plot(loss, label='Train_loss')
plt.plot(val_loss, label='Val_loss')
plt.plot(val_acc, label='Val_acc')
plt.plot(acc, label='Train_acc')
plt.legend()
plt.show()


model.evaluate(val_data, val_labels)

# COMMAND ----------

from src.utils import to_rgb

def get_labels_and_bags(data, labels, bag_class, predictions=None):
    if bag_class == "positive":
        if predictions is not None:
            print(1)
            labels = np.where(predictions.argmax(1) == 1)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]
            return labels, bags
        else:
            print(2)
            labels = np.where(labels == 1)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]
            return labels, bags
    elif bag_class == "negative":
        if predictions is not None:
            print(3)
            labels = np.where(predictions.argmax(1) == 0)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]
            return labels, bags
        else:
            print(4)
            labels = np.where(labels == 0)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]
            return labels, bags

def plot(data, labels, bag_class, predictions=None, attention_weights=None):

    labels, bags = get_labels_and_bags(data, labels, bag_class, predictions=predictions)
    labels = np.array(labels).reshape(-1)



    print(f"The bag class label is {bag_class}")
    for i in range(PLOT_SIZE):
        figure = plt.figure(figsize=(8, 8))
        print(f"Bag number: {labels[i]}")
        for j in range(BAG_SIZE):
            image = bags[j][i]
            figure.add_subplot(1, BAG_SIZE, j + 1)
            plt.grid(False)
            if attention_weights is not None:
                plt.title(np.around(attention_weights[labels[i]][j], 2))

            plt.imshow(to_rgb(image[15]))
        plt.show()


# Plot some of validation data bags per class.


# COMMAND ----------

def predict(data, labels, trained_models):

    # Collect info per model.
    models_predictions = []
    models_attention_weights = []
    models_losses = []
    models_accuracies = []

    for model in trained_models:

        # Predict output classes on data.
        predictions = model.predict(data)
        models_predictions.append(predictions)

        # Create intermediate model to get MIL attention layer weights.
        intermediate_model = keras.Model(model.input, model.get_layer("alpha").output)

        # Predict MIL attention layer weights.
        intermediate_predictions = intermediate_model.predict(data)

        attention_weights = np.squeeze(np.swapaxes(intermediate_predictions, 1, 0))
        models_attention_weights.append(attention_weights)

        loss, accuracy = model.evaluate(data, labels, verbose=0)
        models_losses.append(loss)
        models_accuracies.append(accuracy)

    print(
        f"The average loss and accuracy are {np.sum(models_losses, axis=0):.2f}"
        f" and {100 * np.sum(models_accuracies, axis=0):.2f} % resp."
    )

    return (
        np.sum(models_predictions, axis=0),
        np.sum(models_attention_weights, axis=0),
    )


# Evaluate and predict classes and attention scores on validation data.
class_predictions, attention_params = predict(val_data, val_labels, [model])

# Plot some results from our validation data.
plot(
    val_data,
    val_labels,
    "positive",
    predictions=class_predictions,
    attention_weights=attention_params,
)
plot(
    val_data,
    val_labels,
    "negative",
    predictions=class_predictions,
    attention_weights=attention_params,
)
