import tensorflow as tf 

def gated_linear_layer(inputs, gates, name = None):

    activation = tf.multiply(x = inputs, y = tf.sigmoid(gates), name = name)

    return activation

def instance_norm_layer(
    inputs, 
    epsilon = 1e-06, 
    activation_fn = None, 
    name = None):

    instance_norm_layer = tf.contrib.layers.instance_norm(
        inputs = inputs,
        epsilon = epsilon,
        activation_fn = activation_fn)

    return instance_norm_layer

def conv1d_layer(
    inputs, 
    filters, 
    kernel_size, 
    strides = 1, 
    padding = 'same', 
    activation = None,
    kernel_initializer = None,
    name = None):

    conv_layer = tf.layers.conv1d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        activation = activation,
        kernel_initializer = kernel_initializer,
        name = name)

    return conv_layer

def conv2d_layer(
    inputs, 
    filters, 
    kernel_size, 
    strides, 
    padding = 'same', 
    activation = None,
    kernel_initializer = None,
    name = None):

    conv_layer = tf.layers.conv2d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        activation = activation,
        kernel_initializer = kernel_initializer,
        name = name)

    return conv_layer

def residual1d_block(
    inputs, 
    filters = 1024, 
    kernel_size = 3, 
    strides = 1,
    name_prefix = 'residule_block_'):

    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')
    h2 = conv1d_layer(inputs = h1_glu, filters = filters // 2, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h2_conv')
    h2_norm = instance_norm_layer(inputs = h2, activation_fn = None, name = name_prefix + 'h2_norm')
    
    h3 = inputs + h2_norm

    return h3

def downsample1d_block(
    inputs, 
    filters, 
    kernel_size, 
    strides,
    name_prefix = 'downsample1d_block_'):

    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def downsample2d_block(
    inputs, 
    filters, 
    kernel_size, 
    strides,
    name_prefix = 'downsample2d_block_'):

    h1 = conv2d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv2d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def upsample1d_block(
    inputs, 
    filters, 
    kernel_size, 
    strides,
    shuffle_size = 2,
    name_prefix = 'upsample1d_block_'):
    
    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_shuffle = pixel_shuffler(inputs = h1, shuffle_size = shuffle_size, name = name_prefix + 'h1_shuffle')
    h1_norm = instance_norm_layer(inputs = h1_shuffle, activation_fn = None, name = name_prefix + 'h1_norm')

    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_shuffle_gates = pixel_shuffler(inputs = h1_gates, shuffle_size = shuffle_size, name = name_prefix + 'h1_shuffle_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_shuffle_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')

    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def pixel_shuffler(inputs, shuffle_size = 2, name = None):

    n = tf.shape(inputs)[0]
    w = tf.shape(inputs)[1]
    c = inputs.get_shape().as_list()[2]

    oc = c // shuffle_size
    ow = w * shuffle_size

    outputs = tf.reshape(tensor = inputs, shape = [n, ow, oc], name = name)

    return outputs

class UNet():
    
    # Convolution layer
    @staticmethod
    def conv(inputs, filters, kernel_size=3, activation=tf.nn.relu, l2_reg_scale=None, batchnorm_istraining=None,name = None):
        if l2_reg_scale is None:
            regularizer = None
        else:
            regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
        conved = tf.layers.conv1d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation=activation,
            name = name
        )
        if batchnorm_istraining is not None:
            conved = UNet.bn(conved, batchnorm_istraining)

        return conved
    # Batch normalization
    @staticmethod
    def bn(inputs, is_training):
        normalized = tf.layers.batch_normalization(
            inputs=inputs,
            axis=-1,
            momentum=0.9,
            epsilon=0.001,
            center=True,
            scale=True,
            training=is_training,
        )
        return normalized

    # Pooling layer
    @staticmethod
    def pool(inputs):
        pooled = tf.layers.max_pooling1d(inputs=inputs, pool_size=2, strides=2)
        return pooled

    # Transpose convolution layer (Deconvolution)
    @staticmethod
    def conv_transpose(inputs, filters, l2_reg_scale=None, name = None):
        if l2_reg_scale is None:
            regularizer = None
        else:
            regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
        conved = tf.layers.conv1d_transpose(
            inputs=inputs,
            filters=filters,
            strides=2,
            kernel_size=2,
            padding='same',
            activation=tf.nn.relu,
            kernel_regularizer=regularizer
        )
        return conved

def generator_new(inputs, reuse = False, scope_name = 'generator_gatedcnn', num_features = 24, training = True):

    # inputs has shape [batch_size, num_features, time]
    # we need to convert it to [batch_size, time, num_features] for 1D convolution
    inputs = tf.transpose(inputs, perm = [0, 2, 1], name = 'input_transpose')

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False
            
        is_training = training
        l2_reg = None
        
        conv1_1 = UNet.conv(inputs, filters=64, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv1_2 = UNet.conv(conv1_1, filters=64, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool1 = UNet.pool(conv1_2)

        # 1/2, 1/2, 64
        conv2_1 = UNet.conv(pool1, filters=128, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv2_2 = UNet.conv(conv2_1, filters=128, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool2 = UNet.pool(conv2_2)

        # 1/4, 1/4, 128
        conv3_1 = UNet.conv(pool2, filters=256, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv3_2 = UNet.conv(conv3_1, filters=256, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool3 = UNet.pool(conv3_2)

        # 1/8, 1/8, 256
        conv4_1 = UNet.conv(pool3, filters=512, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv4_2 = UNet.conv(conv4_1, filters=512, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool4 = UNet.pool(conv4_2)
        
        r1 = residual1d_block(inputs = pool4, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block1_')
        r2 = residual1d_block(inputs = r1, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block2_')
        r3 = residual1d_block(inputs = r2, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block3_')
        r4 = residual1d_block(inputs = r3, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block4_')
        r5 = residual1d_block(inputs = r4, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block5_')
        r6 = residual1d_block(inputs = r5, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block6_')
        
        up0 = upsample1d_block(inputs = r6, filters = 1024, kernel_size = 5, strides = 1, shuffle_size = 2, name_prefix="up0")
        concated0 = tf.concat([up0, conv4_2], axis=2, name="concat_0")

        conv_up1_1 = UNet.conv(concated0, filters=512, l2_reg_scale=l2_reg, name="conv_up_11")
        conv_up1_2 = UNet.conv(conv_up1_1, filters=512, l2_reg_scale=l2_reg, name="conv_up_12")
        up1 = upsample1d_block(inputs = conv_up1_2, filters = 512, kernel_size = 5, strides = 1, shuffle_size = 2, name_prefix="up1")
        concated1 = tf.concat([up1, conv3_2], axis=2, name="concat_1")

        conv_up2_1 = UNet.conv(concated1, filters=256, l2_reg_scale=l2_reg, name="conv_up_21")
        conv_up2_2 = UNet.conv(conv_up2_1, filters=256, l2_reg_scale=l2_reg, name="conv_up_22")
        up2 = upsample1d_block(inputs = conv_up2_2, filters = 256, kernel_size = 5, strides = 1, shuffle_size = 2, name_prefix="up2")
        concated2 = tf.concat([up2, conv2_2], axis=2, name="concat_2")

        conv_up3_1 = UNet.conv(concated2, filters=128, l2_reg_scale=l2_reg, name="conv_up_31")
        conv_up3_2 = UNet.conv(conv_up3_1, filters=128, l2_reg_scale=l2_reg, name="conv_up_32")
        up3 = upsample1d_block(inputs = conv_up3_2, filters = 128, kernel_size = 5, strides = 1, shuffle_size = 2, name_prefix="up3")
        concated3 = tf.concat([up3, conv1_2], axis=2, name="concat_3")

        conv_up4_1 = UNet.conv(concated3, filters=64, l2_reg_scale=l2_reg, name="conv_up_41")
        conv_up4_2 = UNet.conv(conv_up4_1, filters=64, l2_reg_scale=l2_reg, name="conv_up_42")
        
        outputs = conv1d_layer(inputs = conv_up4_2, filters = num_features, kernel_size = 15, strides = 1, activation = None)
        outputs = tf.transpose(outputs, perm = [0, 2, 1], name = 'output_transpose')
       
    return outputs

def generator_gatedcnn(inputs, reuse = False, scope_name = 'generator_gatedcnn', num_features = 24, training = True):

    # inputs has shape [batch_size, num_features, time]
    # we need to convert it to [batch_size, time, num_features] for 1D convolution
    inputs = tf.transpose(inputs, perm = [0, 2, 1], name = 'input_transpose')

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv1d_layer(inputs = inputs, filters = 128, kernel_size = 15, strides = 1, activation = None, name = 'h1_conv')
        h1_gates = conv1d_layer(inputs = inputs, filters = 128, kernel_size = 15, strides = 1, activation = None, name = 'h1_conv_gates')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')

        # Downsample
        d1 = downsample1d_block(inputs = h1_glu, filters = 256, kernel_size = 5, strides = 2, name_prefix = 'downsample1d_block1_')
        d2 = downsample1d_block(inputs = d1, filters = 512, kernel_size = 5, strides = 2, name_prefix = 'downsample1d_block2_')

        # Residual blocks
        r1 = residual1d_block(inputs = d2, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block1_')
        r2 = residual1d_block(inputs = r1, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block2_')
        r3 = residual1d_block(inputs = r2, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block3_')
        r4 = residual1d_block(inputs = r3, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block4_')
        r5 = residual1d_block(inputs = r4, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block5_')
        r6 = residual1d_block(inputs = r5, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block6_')
        
        u1 = upsample1d_block(inputs = r6, filters = 1024, kernel_size = 5, strides = 1, shuffle_size = 2, name_prefix = 'upsample1d_block1_')
        u2 = upsample1d_block(inputs = u1, filters = 512, kernel_size = 5, strides = 1, shuffle_size = 2, name_prefix = 'upsample1d_block2_')

        # Output
        o1 = conv1d_layer(inputs = u2, filters = num_features, kernel_size = 15, strides = 1, activation = None, name = 'o1_conv')
        o2 = tf.transpose(o1, perm = [0, 2, 1], name = 'output_transpose')

    return o2
    

def discriminator(inputs, reuse = False, scope_name = 'discriminator'):

    # inputs has shape [batch_size, num_features, time]
    # we need to add channel for 2D convolution [batch_size, num_features, time, 1]
    inputs = tf.expand_dims(inputs, -1)

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [3, 3], strides = [1, 2], activation = None, name = 'h1_conv')
        h1_gates = conv2d_layer(inputs = inputs, filters = 128, kernel_size = [3, 3], strides = [1, 2], activation = None, name = 'h1_conv_gates')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')

        # Downsample
        d1 = downsample2d_block(inputs = h1_glu, filters = 256, kernel_size = [3, 3], strides = [2, 2], name_prefix = 'downsample2d_block1_')
        d2 = downsample2d_block(inputs = d1, filters = 512, kernel_size = [3, 3], strides = [2, 2], name_prefix = 'downsample2d_block2_')
        d3 = downsample2d_block(inputs = d2, filters = 1024, kernel_size = [6, 3], strides = [1, 2], name_prefix = 'downsample2d_block3_')

        # Output
        o1 = tf.layers.dense(inputs = d3, units = 1, activation = tf.nn.sigmoid)

        return o1

