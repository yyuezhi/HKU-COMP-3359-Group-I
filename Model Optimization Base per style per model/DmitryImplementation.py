# This is just copy from the post
# https://subroy13.github.io/post/post3/
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import time
import functools
import PIL.Image
import IPython.display as display
import matplotlib.pyplot as plt

style_path = '../trainingimages/style/style01.jpg'
content_path = '../trainingimages/content/content01.png'

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv2']
content_layers = ['block4_conv2']


def tensor_to_image(tensor):
    tensor = tf.clip_by_value(tensor, clip_value_min=0.0, clip_value_max=255.0)
    tensor = np.array(tensor, dtype=np.uint8)   # convert tf array to np array of integers
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1  # asserts that the BATCH_SIZE = 1
        tensor = tensor[0]   # take the first image
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img, rescale = False):
    # we rescale the image to max dimension 256 for fasters processing
    max_dim = 256    
    img = tf.io.read_file(path_to_img)   # read the image
    img = tf.image.decode_image(img, channels=3)    # decode into image content
    img = tf.image.convert_image_dtype(img, tf.float32)    # convert to float
    
    if rescale:
        img = tf.image.resize(img, tf.constant([max_dim, max_dim]))
    else:
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)   
        # get the shape of image, cast it to float type for division, expect the last channel dimension
        long_dim = max(shape)
        scale = max_dim / long_dim    # scale accordingly
        new_shape = tf.cast(shape * scale, tf.int32)   # cast the new shape to integer
        img = tf.image.resize(img, new_shape)   # resize image
        
    img = img[tf.newaxis, :]   # newaxis builts a new batch axis in the image at first dimension
    return img

def periodic_padding(x, padding=1):
    '''
    x: shape (batch_size, d1, d2)
    return x padded with periodic boundaries. i.e. torus or donut
    '''
    d1 = x.shape[1] # dimension 1: height
    d2 = x.shape[2] # dimension 2: width
    p = padding
    # assemble padded x from slices
    #            tl,tc,tr
    # padded_x = ml,mc,mr
    #            bl,bc,br
    top_left = x[:, -p:, -p:] # top left
    top_center = x[:, -p:, :] # top center
    top_right = x[:, -p:, :p] # top right
    middle_left = x[:, :, -p:] # middle left
    middle_center = x # middle center
    middle_right = x[:, :, :p] # middle right
    bottom_left = x[:, :p, -p:] # bottom left
    bottom_center = x[:, :p, :] # bottom center
    bottom_right = x[:, :p, :p] # bottom right
    top = tf.concat([top_left, top_center, top_right], axis=2)
    middle = tf.concat([middle_left, middle_center, middle_right], axis=2)
    bottom = tf.concat([bottom_left, bottom_center, bottom_right], axis=2)
    padded_x = tf.concat([top, middle, bottom], axis=1)
    return padded_x

def CircularPadding(inputs, kernel_size = 3):
    """Prepares padding for Circular convolution"""
    # split all the filters
    n_filters_in = inputs.shape[-1]
    input_split = tf.split(inputs, n_filters_in, axis = -1)
    output_split = []
    for part in input_split:
        part = tf.squeeze(part, axis = -1)
        outs = periodic_padding(part, padding = int(kernel_size / 2))
        outs = tf.expand_dims(outs, axis = -1)
        output_split.append(outs)
    return tf.concat(output_split, axis = -1)



def conv_block(input_size, in_filters, out_filters):
    """Implements the convolutional block with 3x3, 3x3, 1x1 filters, with proper batch normalization and activation"""
    inputs = tf.keras.layers.Input((input_size, input_size, in_filters, ))   # in_filters many channels of input image
    
    # first 3x3 conv
    conv1_pad = tf.keras.layers.Lambda(lambda x: CircularPadding(x))(inputs)
    conv1_out = tf.keras.layers.Conv2D(out_filters, kernel_size = (3, 3), strides = 1, 
                                       padding = 'valid', name = 'conv1')(conv1_pad)
    hidden_1 = tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")(conv1_out)
    conv1_out_final = tf.keras.layers.LeakyReLU(name = 'rel1')(hidden_1)
    
    # second 3x3 conv
    conv2_pad = tf.keras.layers.Lambda(lambda x: CircularPadding(x))(conv1_out_final)
    conv2_out = tf.keras.layers.Conv2D(out_filters, kernel_size = (3, 3), strides = 1, 
                                       padding = 'valid', name = 'conv2')(conv2_pad)
    hidden_2 = tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")(conv2_out)
    conv2_out_final = tf.keras.layers.LeakyReLU(name = 'rel2')(hidden_2)
    
    # final 1x1 conv
    conv3_out = tf.keras.layers.Conv2D(out_filters, kernel_size = (1, 1), strides = 1, 
                                       padding = 'same', name = 'conv3')(conv2_out_final)
    hidden_3 = tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")(conv3_out)
    conv3_out_final = tf.keras.layers.LeakyReLU(name = 'rel3')(hidden_3)
    
    # final model
    conv_block = tf.keras.models.Model(inputs, conv3_out_final)
    return conv_blo



def join_block(input_size, n_filter_low, n_filter_high):
    input1 = tf.keras.layers.Input((input_size, input_size, n_filter_low, ))  # input to low resolution image
    input2 = tf.keras.layers.Input((2*input_size, 2*input_size, n_filter_high, ))  # input to high resolution image
    upsampled_input = tf.keras.layers.UpSampling2D(size = (2, 2))(input1)
    hidden_1 = tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")(upsampled_input)
    hidden_2 = tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")(input2)
    
    outputs = tf.keras.layers.Concatenate(axis=-1)([hidden_1, hidden_2])
    
    # final model
    join_block = tf.keras.models.Model([input1, input2], outputs)
    return join_block

def generator_network():
    # create input nodes for noise tensors
    noise1 = tf.keras.layers.Input((256, 256, 3, ), name = 'noise_1')
    noise2 = tf.keras.layers.Input((128, 128, 3, ), name = 'noise_2')
    noise3 = tf.keras.layers.Input((64, 64, 3, ), name = 'noise_3')
    noise4 = tf.keras.layers.Input((32, 32, 3, ), name = 'noise_4')
    noise5 = tf.keras.layers.Input((16, 16, 3, ), name = 'noise_5')
    noise6 = tf.keras.layers.Input((8, 8, 3, ), name = 'noise_6')
    content = tf.keras.layers.Input((256, 256, 3, ), name = 'content_input')

    # downsample the content image
    content_image_8 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, tf.constant([8, 8])))(content)
    content_image_16 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, tf.constant([16, 16])))(content)
    content_image_32 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, tf.constant([32, 32])))(content)
    content_image_64 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, tf.constant([64, 64])))(content)
    content_image_128 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, tf.constant([128, 128])))(content)
    
    # create concatenation of downsampled content image and input nodes
    noise6_con = tf.keras.layers.Concatenate(axis=-1)([noise6, content_image_8])
    noise5_con = tf.keras.layers.Concatenate(axis=-1)([noise5, content_image_16])
    noise4_con = tf.keras.layers.Concatenate(axis=-1)([noise4, content_image_32])
    noise3_con = tf.keras.layers.Concatenate(axis=-1)([noise3, content_image_64])
    noise2_con = tf.keras.layers.Concatenate(axis=-1)([noise2, content_image_128])
    noise1_con = tf.keras.layers.Concatenate(axis=-1)([noise1, content])
    
    noise6_conv = conv_block(8, 6, 8)(noise6_con)   # that produces 8x8x8 tensor
    noise5_conv = conv_block(16, 6, 8)(noise5_con)   # that produces 16x16x8 tensor
    join5 = join_block(8, 8, 8)([noise6_conv, noise5_conv])   # that produces 16x16x16 tensor
    
    join5_conv = conv_block(16, 16, 16)(join5)   # produces 16x16x16 tensor
    noise4_conv = conv_block(32, 6, 8)(noise4_con)   # that produces 32x32x8 tensor
    join4 = join_block(16, 16, 8)([join5_conv, noise4_conv])   # produces 32x32x24 tensor
    
    join4_conv = conv_block(32, 24, 24)(join4)   # produces 32x32x24 tensor
    noise3_conv = conv_block(64, 6, 8)(noise3_con)  # that produces 64x64x8 tensor
    join3 = join_block(32, 24, 8)([join4_conv, noise3_conv])   # produces 64x64x32 tensor
    
    join3_conv = conv_block(64, 32, 32)(join3)   # produces 64x64x32 tensor
    noise2_conv = conv_block(128, 6, 8)(noise2_con)  # that produces 128x128x8 tensor
    join2 = join_block(64, 32, 8)([join3_conv, noise2_conv])   # produces 128x128x40 tensor
    
    join2_conv = conv_block(128, 40, 40)(join2)   # produces 128x128x40 tensor
    noise1_conv = conv_block(256, 6, 8)(noise1_con)  # that produces 256x256x8 tensor
    join1 = join_block(128, 40, 8)([join2_conv, noise1_conv])   # produces 256x256x48 tensor
    
    output = conv_block(256, 48, 3)(join1)   # produces 256x256x3 tensor
    
    model = tf.keras.models.Model([content, noise1, noise2, noise3, noise4, noise5, noise6], output, name = 'generator')
    
    return model


def generator_network():
    # create input nodes for noise tensors
    noise1 = tf.keras.layers.Input((256, 256, 3, ), name = 'noise_1')
    noise2 = tf.keras.layers.Input((128, 128, 3, ), name = 'noise_2')
    noise3 = tf.keras.layers.Input((64, 64, 3, ), name = 'noise_3')
    noise4 = tf.keras.layers.Input((32, 32, 3, ), name = 'noise_4')
    noise5 = tf.keras.layers.Input((16, 16, 3, ), name = 'noise_5')
    noise6 = tf.keras.layers.Input((8, 8, 3, ), name = 'noise_6')
    content = tf.keras.layers.Input((256, 256, 3, ), name = 'content_input')

    # downsample the content image
    content_image_8 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, tf.constant([8, 8])))(content)
    content_image_16 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, tf.constant([16, 16])))(content)
    content_image_32 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, tf.constant([32, 32])))(content)
    content_image_64 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, tf.constant([64, 64])))(content)
    content_image_128 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, tf.constant([128, 128])))(content)
    
    # create concatenation of downsampled content image and input nodes
    noise6_con = tf.keras.layers.Concatenate(axis=-1)([noise6, content_image_8])
    noise5_con = tf.keras.layers.Concatenate(axis=-1)([noise5, content_image_16])
    noise4_con = tf.keras.layers.Concatenate(axis=-1)([noise4, content_image_32])
    noise3_con = tf.keras.layers.Concatenate(axis=-1)([noise3, content_image_64])
    noise2_con = tf.keras.layers.Concatenate(axis=-1)([noise2, content_image_128])
    noise1_con = tf.keras.layers.Concatenate(axis=-1)([noise1, content])
    
    noise6_conv = conv_block(8, 6, 8)(noise6_con)   # that produces 8x8x8 tensor
    noise5_conv = conv_block(16, 6, 8)(noise5_con)   # that produces 16x16x8 tensor
    join5 = join_block(8, 8, 8)([noise6_conv, noise5_conv])   # that produces 16x16x16 tensor
    
    join5_conv = conv_block(16, 16, 16)(join5)   # produces 16x16x16 tensor
    noise4_conv = conv_block(32, 6, 8)(noise4_con)   # that produces 32x32x8 tensor
    join4 = join_block(16, 16, 8)([join5_conv, noise4_conv])   # produces 32x32x24 tensor
    
    join4_conv = conv_block(32, 24, 24)(join4)   # produces 32x32x24 tensor
    noise3_conv = conv_block(64, 6, 8)(noise3_con)  # that produces 64x64x8 tensor
    join3 = join_block(32, 24, 8)([join4_conv, noise3_conv])   # produces 64x64x32 tensor
    
    join3_conv = conv_block(64, 32, 32)(join3)   # produces 64x64x32 tensor
    noise2_conv = conv_block(128, 6, 8)(noise2_con)  # that produces 128x128x8 tensor
    join2 = join_block(64, 32, 8)([join3_conv, noise2_conv])   # produces 128x128x40 tensor
    
    join2_conv = conv_block(128, 40, 40)(join2)   # produces 128x128x40 tensor
    noise1_conv = conv_block(256, 6, 8)(noise1_con)  # that produces 256x256x8 tensor
    join1 = join_block(128, 40, 8)([join2_conv, noise1_conv])   # produces 256x256x48 tensor
    
    output = conv_block(256, 48, 3)(join1)   # produces 256x256x3 tensor
    
    model = tf.keras.models.Model([content, noise1, noise2, noise3, noise4, noise5, noise6], output, name = 'generator')
    
    return model

def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')  # load the vgg model
    vgg.trainable = False    # do not train over vgg model parameters
  
    outputs = [vgg.get_layer(name).output for name in layer_names]    # the output of the layers that we want

    model = tf.keras.Model([vgg.input], outputs)   # create a keras model
    return model


class TextureNetwork(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(TextureNetwork, self).__init__()   # initialize the superClass
        self.vgg =  vgg_layers(style_layers + content_layers)    # obtain a VGG19 model with outputs being the style and content layers
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False  # we are not going to train vgg network

        self.gen = generator_network()   # create a generator network as part of it
        self.gen.trainable = True   # we are going to train this generator
        

    def call(self, content, batch_size = 16):
        # generates noise required for the network
        noise1 = tf.random.uniform((batch_size, 256, 256, 3))
        noise2 = tf.random.uniform((batch_size, 128, 128, 3))
        noise3 = tf.random.uniform((batch_size, 64, 64, 3))
        noise4 = tf.random.uniform((batch_size, 32, 32, 3))
        noise5 = tf.random.uniform((batch_size, 16, 16, 3))
        noise6 = tf.random.uniform((batch_size, 8, 8, 3))
    
        gen_image = self.gen([content, noise1, noise2, noise3, noise4, noise5, noise6])   # pass through the generator to obtain generated image
    
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(gen_image)  # preprocess the image
        outputs = self.vgg(preprocessed_input)  # get the output from only the required layers
        
        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                      outputs[self.num_style_layers:])
        
        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]  # create style type output to compare

        style_dict = {style_name:value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        content_dict = {content_name:value 
                    for content_name, value 
                    in zip(self.content_layers, content_outputs)}


        return {'gen':gen_image, 'content':content_dict, 'style':style_dict}


def extract_targets(inputs):
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)  # preprocess the input image
    outputs = vgg_layers(style_layers + content_layers)(preprocessed_input)  # get the output from only the required layers
        
    style_outputs, content_outputs = (outputs[:len(style_layers)], 
                                       outputs[len(style_layers):])
        
    style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]  # create style type output to compare

    style_dict = {style_name:value
                      for style_name, value
                      in zip(style_layers, style_outputs)}

    content_dict = {content_name:value 
                    for content_name, value 
                    in zip(content_layers, content_outputs)}

    return {'content':content_dict, 'style':style_dict}


def custom_loss(outputs, batch_size):
    gen_outputs = outputs['gen']
    style_outputs = outputs['style']   # for generated image, get the style
    content_outputs = outputs['content']  # get content
    batch_loss = 0
    for i in range(batch_size):
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name][i]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
        style_loss *= style_weight / len(style_layers)

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name][i]-content_targets[name])**2) 
                                 for name in content_outputs.keys()])
        content_loss *= content_weight / len(content_layers)
        
        loss = style_loss + content_loss
        batch_loss += loss
        
    batch_loss /= batch_size
    return batch_loss

@tf.function()
def train_step(content_image, batch_size):
    
    with tf.GradientTape() as tape:
        outputs = tex_net(content_image, batch_size)
        loss = custom_loss(outputs, batch_size)
        
    gradients = tape.gradient(loss, tex_net.trainable_variables)  # obtain the gradients recorded by the tape
    optimizer.apply_gradients(zip(gradients, tex_net.trainable_variables))   # apply the training rule using the gradients to modify the current value of prameters
    return output, loss


def main():
    content_image = load_img(content_path, rescale = True)
    tensor_to_image(content_image * 255.0)

    style_image = load_img(style_path, rescale = False)
    tensor_to_image(style_image * 255.0)

    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)  # use an Adam optimizer
    tex_net = TextureNetwork(style_layers, content_layers)   # create the texture network
    output = tex_net(content_image, 1)

    style_targets = extract_targets(style_image)['style']
    content_targets = extract_targets(content_image)['content']


    batch_size = 32
    my_content = tf.concat([content_image for _ in range(batch_size)], axis = 0)

    n_epoch = 10
    n_iter = 250
    iter_to_show_output = 25

    loss_array = []
    for epoch in range(n_epoch):
        msg = 'Epoch: ' + str(epoch)
        print(msg)
        os.system('echo ' + msg)
        for step in range(n_iter):
            outputs, loss = train_step(my_content, batch_size)
            if step % iter_to_show_output == 0:
                os.system('echo loss: ' + str(float(loss)))
                print('Loss: ', loss)
                loss_array.append(loss)
        display.display(tensor_to_image(tex_net(content_image, 1)['gen']))

if __name__ == '__main__':
    main()