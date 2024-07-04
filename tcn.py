import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv1D, SpatialDropout1D, LayerNormalization, BatchNormalization

class TCN(Layer):
    def __init__(self, nb_filters=64, kernel_size=2, nb_stacks=1, dilations=None,
                 use_skip_connections=True, padding='causal', use_batch_norm=False,
                 use_layer_norm=False, use_weight_norm=False, dropout_rate=0.0,
                 return_sequences=False, activation='relu', kernel_initializer='he_normal',
                 use_conv_bias=True, **kwargs):

        # Call parent constructor
        super(TCN, self).__init__(**kwargs)

        # Attributes
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.nb_stacks = nb_stacks
        self.padding = padding
        self.dilations = dilations if dilations is not None else [1, 2, 4, 8, 16, 32]
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.use_conv_bias = use_conv_bias
        
        self.supports_masking = True
        self.build_output_shape = None

    def build(self, input_shape):
        self.build_output_shape = input_shape

        input_shape_list = list(input_shape)
        
        # Assertions
        assert len(input_shape_list) == 3
        assert input_shape_list[2] > 0

        # Layers
        self.residual_blocks = []
        for s in range(self.nb_stacks):
            for d in self.dilations:
                self.residual_blocks.append(ResidualBlock(
                    dilation_rate=d,
                    nb_filters=self.nb_filters,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    activation=self.activation,
                    use_batch_norm=self.use_batch_norm,
                    use_layer_norm=self.use_layer_norm,
                    use_weight_norm=self.use_weight_norm,
                    dropout_rate=self.dropout_rate,
                    kernel_initializer=self.kernel_initializer,
                    use_conv_bias=self.use_conv_bias
                ))
        
        # Build all the layers
        for layer in self.residual_blocks:
            layer.build(input_shape)
        
        # Final layer
        self.final_conv = Conv1D(filters=self.nb_filters, kernel_size=1, padding='same')
        self.final_conv.build(input_shape)
        
        # Output layer
        if not self.return_sequences:
            output_size = input_shape_list[1]
            self.slicer_layer = layers.Lambda(lambda tt: tt[:, -output_size:, :])
            self.slicer_layer.build(input_shape_list)

        # Call the parent's build method
        super(TCN, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        for layer in self.residual_blocks:
            x = layer(x)
        x = self.final_conv(x)
        return x if self.return_sequences else self.slicer_layer(x)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return input_shape
        else:
            return (input_shape[0], self.nb_filters)

class ResidualBlock(Layer):
    def __init__(self, dilation_rate, nb_filters, kernel_size,
                 padding, activation='relu', use_batch_norm=False,
                 use_layer_norm=False, use_weight_norm=False,
                 dropout_rate=0, kernel_initializer='he_normal',
                 use_conv_bias=True, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer
        self.use_conv_bias = use_conv_bias

    def build(self, input_shape):
        self.conv1 = Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size,
                            dilation_rate=self.dilation_rate, padding=self.padding,
                            kernel_initializer=self.kernel_initializer,
                            use_bias=self.use_conv_bias)
        self.conv2 = Conv1D(filters=self.nb_filters, kernel_size=self.kernel_size,
                            dilation_rate=self.dilation_rate, padding=self.padding,
                            kernel_initializer=self.kernel_initializer,
                            use_bias=self.use_conv_bias)
        
        if self.use_batch_norm:
            self.bn1 = BatchNormalization()
            self.bn2 = BatchNormalization()
        
        if self.use_layer_norm:
            self.ln1 = LayerNormalization()
            self.ln2 = LayerNormalization()
        
        if self.dropout_rate > 0:
            self.dropout1 = SpatialDropout1D(rate=self.dropout_rate)
            self.dropout2 = SpatialDropout1D(rate=self.dropout_rate)
        
        self.conv1.build(input_shape)
        self.conv2.build(input_shape)
        
        super(ResidualBlock, self).build(input_shape)

    def call(self, inputs):
        x = self.conv1(inputs)
        if self.use_batch_norm:
            x = self.bn1(x)
        if self.use_layer_norm:
            x = self.ln1(x)
        x = tf.keras.layers.Activation(self.activation)(x)
        if self.dropout_rate > 0:
            x = self.dropout1(x)
        
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        if self.use_layer_norm:
            x = self.ln2(x)
        x = tf.keras.layers.Activation(self.activation)(x)
        if self.dropout_rate > 0:
            x = self.dropout2(x)
        
        return tf.keras.layers.Add()([x, inputs])

    def compute_output_shape(self, input_shape):
        return input_shape
