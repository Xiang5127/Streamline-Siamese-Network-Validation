import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.applications import MobileNetV2

IMAGE_SIZE = (224, 224)
EMBEDDING_DIM = 128


class EuclideanDistance(layers.Layer):
    """Computes the Euclidean (L2) distance between two embedding vectors."""

    def call(self, inputs):
        a, b = inputs
        squared_diff = tf.square(a - b)
        sum_squared = tf.reduce_sum(squared_diff, axis=1, keepdims=True)
        # epsilon guards against sqrt(0) gradients
        return tf.sqrt(tf.maximum(sum_squared, tf.keras.backend.epsilon()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)


def build_embedding_network(input_shape=(224, 224, 3), embedding_dim=EMBEDDING_DIM):
    """MobileNetV2 (frozen) -> GAP -> Dense(128) -> L2-normalized embedding."""
    backbone = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    backbone.trainable = False

    inputs = Input(shape=input_shape, name="embedding_input")
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(embedding_dim, name="embedding_dense")(x)
    outputs = layers.Lambda(
        lambda t: tf.math.l2_normalize(t, axis=1),
        name="l2_normalize",
        output_shape=(embedding_dim,),
    )(x)

    return Model(inputs, outputs, name="embedding_network")


def build_siamese_model(input_shape=(224, 224, 3), embedding_dim=EMBEDDING_DIM):
    """Twin-tower Siamese model using the Functional API."""
    embedding_network = build_embedding_network(input_shape, embedding_dim)

    reference_image = Input(shape=input_shape, name="reference_image")
    live_image = Input(shape=input_shape, name="live_image")

    # Shared weights: same embedding_network instance called twice.
    ref_embedding = embedding_network(reference_image)
    live_embedding = embedding_network(live_image)

    distance = EuclideanDistance(name="euclidean_distance")(
        [ref_embedding, live_embedding]
    )

    return Model(
        inputs={"reference_image": reference_image, "live_image": live_image},
        outputs=distance,
        name="siamese_network",
    )


if __name__ == "__main__":
    model = build_siamese_model()
    print("\n=== Embedding Sub-Network ===")
    model.get_layer("embedding_network").summary()
    print("\n=== Master Siamese Model ===")
    model.summary()
