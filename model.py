import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers


class ConvBlock(layers.Layer):

    def __init__(self, out_channels, strides, padding, dropout, is_final_conv=False):
        super().__init__()

        # Conv 1
        self.conv1 = layers.Conv2D(
            out_channels, (3, 3), 
            strides=strides,
            padding=padding, 
            use_bias=False, 
            activation = "relu",
            kernel_initializer="he_uniform", 
            kernel_regularizer=regularizers.L2()
        )

        # Conv 2
        self.conv2 = layers.Conv2D(
            out_channels, (3, 3), 
            strides= strides,
            padding=padding, 
            use_bias=False, 
            activation = "relu",
            kernel_initializer="he_uniform", 
            kernel_regularizer=regularizers.L2()
        )

        self.dropout = layers.Dropout(dropout)
        self.batch_norm = layers.BatchNormalization()

        self.is_final_conv = is_final_conv

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.dropout(x, training=training)
        x = self.batch_norm(x, training=training)

        if self.is_final_conv:
            return x
        
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

    def __init__(self, feature_fraction: int=1, strides=1, padding="same", dropout=0.5):
        super().__init__()
        assert feature_fraction in range(1, 65, 2), "He so ngoai tam, feature_fraction nam trong [1, 64] va chia het cho 2"
        assert padding in ["same", "valid"], "padding = \"same\" hoac \"valid\""

        features = [64, 128, 256, 512]
        features = [value // feature_fraction for value in features] # Lay so nguyen

        self.pool = layers.MaxPooling2D(2, 2)

        # Encoder
        self.down1 = ConvBlock(features[0], strides, padding, dropout)
        self.down2 = ConvBlock(features[1], strides, padding, dropout)
        self.down3 = ConvBlock(features[2], strides, padding, dropout)
        self.down4 = ConvBlock(features[3], strides, padding, dropout)
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], strides, padding, dropout)

        # Decoder
        self.convT1 = ConvT(features[3], dropout)
        self.up1 = ConvBlock(features[3], strides, padding, dropout)

        self.convT2 = ConvT(features[2], dropout)
        self.up2 = ConvBlock(features[2], strides, padding, dropout)


        self.convT3 = ConvT(features[1], dropout)
        self.up3 = ConvBlock(features[1], strides, padding, dropout)

        self.convT4 = ConvT(features[0], dropout)
        self.up4 = ConvBlock(features[0], strides, padding, dropout)

        # Conv cuoi
        self.final_conv = ConvBlock(2, strides, padding, dropout, True) # Bo ham sigmoid se tinh nhanh hon phan nao

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

        return x

    def copy_crop(self, down, x):
        pass


if __name__ == "__main__":
    inputs = keras.Input(shape=(572, 572, 1))
    test_data = tf.random.normal(shape=(2, 572, 572, 1))

    # Test ConvBlock
    layer_doubleconv = ConvBlock(10, 1, "same", 0.5)
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

    # Test UNet 1
    layer_unet1 = UNet(feature_fraction=1, strides=1, padding="same", dropout=0.5)
    outputs_unet1 = layer_unet1(inputs)

    model_unet1 = keras.Model(inputs, outputs_unet1)
    result_unet1 = model_unet1(test_data)
    print(result_unet1.shape)

    # Test UNet 2
    layer_unet2 = UNet(feature_fraction=1, strides=1, padding="valid", dropout=0.5)
    outputs_unet2 = layer_unet2(inputs)

    model_unet2 = keras.Model(inputs, outputs_unet2)
    result_unet2 = model_unet2(test_data)
    print(result_unet2.shape)

    print("\n\nSuccess\n\n")
                         