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
            kernel_size=(2, 2),
            strides=(2, 2),
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
        self.bottleneck = ConvBlock(features[3], strides, padding, dropout)

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
        x = self.pool(down1) # 284, 284, 64

        down2 = self.down2(x)
        x = self.pool(down2) # 140, 140, 128

        down3 = self.down3(x)
        x = self.pool(down3) # 68, 68, 256

        down4 = self.down4(x)
        x = self.pool(down4) # 32, 32, 512

        # Bottleneck
        x = self.bottleneck(x) # 28, 28, 512
        
        # Decoder
        x = self.convT1(x) # 28, 28, 512
        x = self.copy_crop(x, down4) # 56, 56, 1024
        x = self.up1(x) # 52, 52, 512

        x = self.convT2(x) # 104, 104, 256
        x = self.copy_crop(x, down3) # 104, 104, 512
        x = self.up2(x) # 100, 100, 256

        x = self.convT3(x) # 200, 200, 128
        x = self.copy_crop(x, down2) # 200, 200, 256
        x = self.up3(x) # 196, 196, 128

        x = self.convT4(x) # 392, 392, 64
        x = self.copy_crop(x, down1) # 392, 392, 128
        x = self.up4(x) # 388, 388, 64

        # Final conv
        x = self.final_conv(x) # 386, 386, 2

        return x

    def copy_crop(self, x, down):
        w = x.shape[1]
        h = x.shape[2]
        down = down[:, :w, :h, :]
        x = tf.concat([x, down], 3)

        return x


if __name__ == "__main__":
    inputs = keras.Input(shape=(572, 572, 1))
    test_data = tf.random.normal(shape=(2, 572, 572, 1))

    # Test ConvBlock
    print("\n\nConvBlock test")
    layer_doubleconv = ConvBlock(10, 1, "same", 0.5)
    outputs_convblock = layer_doubleconv(inputs)

    model_convblock = keras.Model(inputs, outputs_convblock)
    result_convblock = model_convblock(test_data)
    print(result_convblock.shape)

    # Test ConvT
    print("\n\nConvT test")
    inputs_convT = keras.Input(shape=(10, 10, 1))
    layer_convT = ConvT(12, 0.5)
    outputs_convT = layer_convT(inputs_convT)

    data_convT = tf.random.normal(shape=(2, 10, 10, 1))
    model_convT = keras.Model(inputs_convT, outputs_convT)
    result_convT = model_convT(data_convT)
    print(result_convT.shape)

    #Test UNet 1
    print("\n\nTest UNet 1, padding = 0")
    layer_unet1 = UNet(feature_fraction=1, strides=1, padding="same", dropout=0.5)
    outputs_unet1 = layer_unet1(inputs)

    model_unet1 = keras.Model(inputs, outputs_unet1)
    result_unet1 = model_unet1(test_data)
    print(result_unet1.shape)

    # Test UNet 2
    print("\n\nTest UNet 2, padding = 1")
    layer_unet2 = UNet(feature_fraction=1, strides=1, padding="valid", dropout=0.5)
    outputs_unet2 = layer_unet2(inputs)

    model_unet2 = keras.Model(inputs, outputs_unet2)
    result_unet2 = model_unet2(test_data)
    print(result_unet2.shape)

    print("\n\nSuccess\n\n")
                         