# keras_mnist_tpu
A demo to show how to use the TPU in google colab.

It is almost the same as training a normal keras model, except that
you need to use `tf.contrib.tpu.keras_to_tpu_model` to transfer
the model to TPU:
```
tpu_model = tf.contrib.tpu.keras_to_tpu_model(
    model,
    strategy=tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(tpu=TPU_ADDRESS)
    )
)
tpu_model.compile(
    optimizer=tf.train.AdamOptimizer(learning_rate=1e-3, ),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['sparse_categorical_accuracy']
)
```

You can find more info in the code or checkout the [colab notebook](https://colab.research.google.com/drive/1MxJcAYBUmcV2Hgrrh7jazSB6dj8Q2oZp#scrollTo=b36wA1EqKwi1)