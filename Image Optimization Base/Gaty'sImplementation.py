from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import tensorflow as tf
import datetime



# Hyperparameter

NOISE_RATIO = 0.5
CONTENT_LAYERS = ['block5_conv2']
STYLE_LAYERS = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']
EPOCH = 10
STEP = 100 # STEP per epoch

LR = 0.1# learning rate for optimizer "Adam"
EPSILON = 1e-1 # 
BETA = 0.99

CONTENT_WEIGHTS = 1e4
STYLE_WEIGHTS = 0.1
VARIATION_WEIGHTS = 30

IMAGE_NUM = 3

#Total Loss keep tracking
CONTENT_LOSS = []
STYLE_LOSS = []
TOTAL_LOSS = []

class NRTmodel(tf.keras.models.Model):
    def __init__(self):
        super(NRTmodel, self).__init__()
        self.vgg = build_vgg()
        self.vgg.trainable = False

    def __call__(self, inputs):
        outputs = self.vgg(inputs)
        style_outputs, content_outputs = (outputs[:len(STYLE_LAYERS)], 
                                      outputs[len(STYLE_LAYERS):])
        
        content_collection = {}
        for i,value in enumerate(content_outputs):
            content_collection[CONTENT_LAYERS[i]] = value
        
        style_collection = {}
        for i,value in enumerate(style_outputs):
            style_collection[STYLE_LAYERS[i]] = value
    
        return {'content':content_collection, 'style':style_collection}
    

def build_vgg():
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in STYLE_LAYERS]+ [vgg.get_layer(name).output for name in CONTENT_LAYERS]
        model = tf.keras.Model([vgg.input], outputs)
        return model


    


def load_image():
    import os 
    curr_dir = "./"
    content_dir = os.path.join(curr_dir,"images","content","") #./images/content/ 
    style_dir = os.path.join(curr_dir,"images","style","") 

    # Get image paths in data_dir
    content_data_paths = os.listdir(content_dir) 
    content_data_paths_full = [ os.path.join(content_dir, p) for p in content_data_paths]

    style_data_paths = os.listdir(style_dir) 
    style_data_paths_full = [ os.path.join(style_dir, p) for p in style_data_paths ]


    # As the number of data is lower, so we do not really need image generator
    content_batch = []
#     content_batch2 = []
    for path in content_data_paths_full :
        if 'checkpoints' not in path:
            image_tensor = tf.keras.preprocessing.image.load_img(path, #read the image from png to img form
              target_size=(224,224),
              interpolation='nearest'
            )
            image_tensor = tf.keras.preprocessing.image.img_to_array(image_tensor,dtype = "float32")  #convert img form to tensor form
            content_batch.append(image_tensor)
            
#             image = tf.io.read_file(path)
#             image = tf.image.decode_image(image, channels=3)
#             image = tf.image.convert_image_dtype(image, tf.float32)
#             image = tf.image.resize(image, [224,224])
#             content_batch2.append(image)

    content_batch = np.array(content_batch)   # Convert the list of tensor to one big tensor 
    content_image = content_batch.reshape((IMAGE_NUM,224,224,3))
   

    
#     print(content_image.shape)
#     print(content_image2.shape)
#     print(content_image)
#     print(content_image2*255)
    # 
    style_batch = []
    for path in style_data_paths_full :
        if 'checkpoints' not in path:
            image_tensor = tf.keras.preprocessing.image.load_img(path, #read the image from png to img form
                target_size=(224,224),
                color_mode='rgb',
                interpolation='nearest')
            image_tensor = tf.keras.preprocessing.image.img_to_array(image_tensor,dtype = "float32")  #convert img form to tensor form
            style_batch.append(image_tensor)


    style_batch = np.array(style_batch)   # Convert the list of tensor to one big tensor 
    style_image = style_batch.reshape((IMAGE_NUM,224,224,3))
#     style_image_vgg = tf.keras.applications.vgg16.preprocess_input(content_style)) # reshape the tensor to desire shape

    #Generates a noisy image by adding random noise to the content_image
    noise_image = np.random.uniform(0, 255, (IMAGE_NUM, 224, 224, 3)).astype('float32')
    noise_image = noise_image*NOISE_RATIO+content_image*(1-NOISE_RATIO)


    w=10
    h=10
    fig=plt.figure(figsize=(12, 12))
    columns = 2
    rows = 2

    fig.add_subplot(rows, columns, 1)
    plt.imshow(tf.cast(content_image[0],tf.int16))
    fig.add_subplot(rows, columns, 2)
    plt.imshow(tf.cast(content_image[1],tf.int16))
    plt.show()
                                        
    return content_image, style_image,noise_image


def total_variation_loss(image):
  x_deltas, y_deltas = high_pass_x_y(image)
  return tf.reduce_mean(x_deltas**2) + tf.reduce_mean(y_deltas**2)

def high_pass_x_y(image):
  x_var = image[:,:,1:,:] - image[:,:,:-1,:]
  y_var = image[:,1:,:,:] - image[:,:-1,:,:]

  return x_var, y_var



def calculate_loss(outputs,target_content,target_style,image):
    output_content = outputs["content"]
    output_style = outputs["style"]
    zeros = [0]*IMAGE_NUM
    content_loss = tf.constant(zeros, shape=[IMAGE_NUM, 1],dtype= 'float32')
    
    
    #Calculate content loss
    content_loss = tf.math.add_n([tf.reduce_mean((tf.reshape(target_content[name],[IMAGE_NUM,-1]) - tf.reshape(output_content[name],[IMAGE_NUM,-1]))**2,keepdims = True,axis = 1)  
                                      for name in output_content.keys()])
    content_loss *= (CONTENT_WEIGHTS / len(CONTENT_LAYERS))
    
    
    
    
    style_loss = tf.constant(zeros, shape=[IMAGE_NUM, 1],dtype= 'float32')
    for name in STYLE_LAYERS:
        m,H ,W, C = output_style[name].get_shape().as_list()
        unrolled_target_style = tf.reshape(target_style[name],[m,-1,C])
        gram_target = tf.matmul(unrolled_target_style, unrolled_target_style, transpose_a=True)/tf.cast(H*W, tf.float32)

        unrolled_output_style = tf.reshape(output_style[name],[m,-1,C])
        gram_output = tf.matmul(unrolled_output_style, unrolled_output_style, transpose_a=True)/tf.cast(H*W, tf.float32)
        style_loss += tf.reduce_mean((tf.reshape(gram_target,[IMAGE_NUM,-1]) - tf.reshape(gram_output,[IMAGE_NUM,-1]))**2,keepdims = True,axis = 1)
        
                
    style_loss *= (STYLE_WEIGHTS / len(STYLE_LAYERS))
    
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]
    variation_loss = (tf.reduce_mean(tf.reshape(x_var,[IMAGE_NUM,-1])**2,keepdims = True,axis = 1) + tf.reduce_mean(tf.reshape(y_var,[IMAGE_NUM,-1])**2,keepdims = True,axis = 1)) * VARIATION_WEIGHTS
    total_loss = tf.add_n([content_loss,style_loss,variation_loss])
        
    return total_loss
#         target_result = tf.linalg.einsum('bijc,bijd->bcd', target_style[name], target_style[name])
#         input_shape = tf.shape(target_style[name])
#         num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
#         target_result = target_result/(num_locations)
          
#         out = tf.linalg.einsum('bijc,bijd->bcd', output_style[name], output_style[name])
#         input_shape = tf.shape(target_style[name])
#         num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
#         out = out/(num_locations)
#         style_loss += tf.add_n([tf.reduce_mean((out-target_result)**2) 
#                            for name in output_style.keys()])
          

def pixel_clip(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255)

def train_one_step(image_tensor,target_content,target_style,backbone_Net,optimizer):
    image = tf.Variable(image_tensor,dtype = 'float32')
    with tf.GradientTape() as tape:
        outputs = backbone_Net(image)
        loss = calculate_loss(outputs,target_content,target_style,image)
    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(pixel_clip(image))
    return image

def main():
    #load the image
    content_tensor, style_tensor, result_tensor= load_image()

    #define the VGG model and then specify the target_content and target_style used
    backbone_Net = NRTmodel()
    target_content = backbone_Net(content_tensor)["content"]
    target_style = backbone_Net(style_tensor)["style"]
    

#     print("  shape: ", result_tensor.shape)
#     print("  min: ", result_tensor.min())
#     print("  max: ", result_tensor.max())
#     print("  mean: ", result_tensor.mean())
#     w=10
#     h=10
#     fig=plt.figure(figsize=(12, 12))
#     columns = 2
#     rows = 2


    
    # #define the optimizer
    optimizer = tf.optimizers.Adam(learning_rate=LR)

    # train the result tensor picture
    for n in range(EPOCH):
        for m in range(STEP):
            result_tensor = train_one_step(result_tensor,target_content,target_style,backbone_Net,optimizer)
        for i in range(IMAGE_NUM):
            store_tensor = tf.cast(result_tensor,tf.int16)
            store_tensor = tf.cast(store_tensor,tf.uint8)
            img = tf.io.encode_jpeg(store_tensor[i])
            tf.io.write_file("picture"+str(i+1)+"_interation"+str(n+1)+".jpg",img)
    w=10
    h=10
    fig=plt.figure(figsize=(12, 12))
    columns = 2
    rows = 3

    

    fig.add_subplot(rows, columns, 1)
    plt.imshow(tf.cast(result_tensor[0],tf.int16))
    fig.add_subplot(rows, columns, 2)
    plt.imshow(tf.cast(result_tensor[1],tf.int16))
#     fig.add_subplot(rows, columns, 3)
#     plt.plot(CONTENT_LOSS)
#     plt.ylabel('content loss')
#     fig.add_subplot(rows, columns, 4)
#     plt.plot(STYLE_LOSS)
#     plt.ylabel('style loss')
#     fig.add_subplot(rows, columns, 5)
#     plt.plot(TOTAL_LOSS)
#     plt.ylabel('total loss')
#     plt.show()
    
    
if __name__ == "__main__":
    main()
