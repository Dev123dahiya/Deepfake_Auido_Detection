import tensorflow as tf
from tensorflow.keras import layers, models


class AttentionLayer(layers.Layer):
    def __init__(self, attention_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], self.attention_dim),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(self.attention_dim,),
            initializer="zeros",
            trainable=True,
        )
        self.u = self.add_weight(
            name="attention_context",
            shape=(self.attention_dim,),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        uit = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        ait = tf.nn.softmax(ait, axis=1)
        ait = tf.expand_dims(ait, axis=-1)
        weighted_input = x * ait
        return tf.reduce_sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update({"attention_dim": self.attention_dim})
        return config


def create_cnn_attention_model(input_shape, num_classes=2):
    inputs = layers.Input(shape=input_shape)
    x = layers.Reshape((*input_shape, 1))(inputs)

    for filters in [64, 128, 256, 512]:
        x = layers.Conv2D(filters, (3, 3), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.RepeatVector(10)(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.3))(x)
    x = AttentionLayer(attention_dim=256, name="attention_layer")(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs=inputs, outputs=outputs)

