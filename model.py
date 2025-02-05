import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers


class ConvBlock(layers.Layer):

    def __init__(self, out_channels, dropout):
        super().__init__()

        # Conv 1
        self.conv1 = layers.Conv2D(
            out_channels, (3, 3), 
            padding="same", 
            use_bias=False, 
            activation = "relu",
            kernel_initializer="he_uniform", 
            kernel_regularizer=regularizers.L2()
        )

        # Conv 2
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
        

class ConvT(layers.Layer):

    def __init__(self, out_channels, dropout):
        super().__init__()

        self.convT = layers.Conv2DTranspose(
            out_channels,
            (2, 2),
            padding="same", 
            use_bias=False, 
            activation = "relu",
            kernel_initializer="he_uniform", 
            kernel_regularizer=regularizers.L2()
        )

        self.dropout = layers.Dropout(dropout)
        self.batch_norm = layers.BatchNormalization()
    
    def call(self, inputs, training=False):
        x = self.convT(inputs)
        x = self.dropout(x, training=training)
        x = self.batch_norm(x)

        return x

class UNet(layers.Layer):

    def __init__(self, feature_fraction: int=1, dropout=0.5):
        assert feature_fraction in range(1, 65, 2), "He so ngoai tam, feature_fraction nam trong [1, 64] va chia het cho 2"

        features = [64, 128, 256, 512]
        features = [value / feature_fraction for value in features]

        self.pool = keras.MaxPooling2D(2, 2)

        # Encoder
        self.down1 = ConvBlock(features[0], dropout)
        self.down2 = ConvBlock(features[1], dropout)
        self.down3 = ConvBlock(features[2], dropout)
        self.down4 = ConvBlock(features[3], dropout)
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], dropout)

        # Decoder
        self.up1 = ConvBlock(features[3], dropout)
        self.up2 = ConvBlock(features[2], dropout)
        self.up3 = ConvBlock(features[1], dropout)
        self.up4 = ConvBlock(features[0], dropout)

    def call(self, inputs):
        # Encoder
        down1 = self.down1(inputs)
        x = self.pool(down1)

        down2 = self.down2(x)
        x = self.pool(down2)

        down3 = self.down3(x)
        x = self.pool(down3)

        down4 = self.down4(x)
        x = self.pool(down4)


if __name__ == "__main__":
    inputs = keras.Input(shape=(224, 224, 1))
    test_data = tf.random.normal(shape=(2, 224, 224, 1))

    # Test ConvBlock
    layer_doubleconv = ConvBlock(10, 0.5)
    outputs_convblock = layer_doubleconv(inputs)

    model_convblock = keras.Model(inputs, outputs_convblock)
    result_convblock = model_convblock(test_data)
    print(result_convblock.shape)

    # Test ConvT
    layer_convT = ConvT(12, 0.5)
    outputs_convT = layer_convT(inputs)

    model_convT = keras.Model(inputs, outputs_convT)
    result_convT = model_convT(test_data)
    print(result_convT.shape)

    print("\n\nSuccess\n\n")
                         