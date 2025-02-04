import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers


class ConvBlock(layers.Layer):

    def __init__(self, out_channels, dropout):
        super().__init__()
        self.conv1 = layers.Conv2D(
            out_channels, (3, 3), 
            padding="same", 
            use_bias=False, 
            activation = "relu",
            kernel_initializer="he_uniform", 
            kernel_regularizer=regularizers.L2()
        )

        self.conv2 = layers.Conv2D(
            out_channels, (3, 3), 
            padding="same", 
            use_bias=False, 
            activation = "relu",
            kernel_initializer="he_uniform", 
            kernel_regularizer=regularizers.L2()
        )

        self.dropout = layers.Dropout(dropout)
        self.batch_norm = layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.dropout(x, training=training)
        x = self.batch_norm(x, training=training)
        
        x = self.conv2(x)
        x = self.dropout(x, training=training)
        x = self.batch_norm(x, training=training)

        return x
        

class UNet(layers.Layer):

    def __init__(self):
        pass

    def call(self):
        pass

if __name__ == "__main__":
    # Test ConvBlock
    test_convblock = tf.random.normal(shape=(2, 224, 224, 1))
    
    inputs = layers.Input(shape=(224, 224, 1))
    doubleconv_layer = ConvBlock(10, 0.5)
    outputs = doubleconv_layer(inputs)
    blockModel = keras.Model(inputs, outputs, name="test_double_conv")

    result_convblock = doubleconv_layer(test_convblock)
    print(result_convblock.shape)
    

    print("\n\nSuccess\n\n")
                         