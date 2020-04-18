# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:13:51 2020

@author: 潘慧杰
"""

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import utils








class encoder(tf.keras.models.Model):
    def __init__(self):
        super(encoder, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32,[9,9],strides=[1,1], padding="SAME",activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64,[3,3],strides=[2,2], padding="SAME",activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(128,[3,3],strides=[2,2], padding="SAME",activation='relu')
        self.instance_norm1 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")
        self.instance_norm2 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")
        self.instance_norm3 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")
    def call(self,inputs):
        #the input channel size is 3 (defult)
        x = self.conv1(inputs)
        x = self.instance_norm1(x)
        x = self.conv2(x)
        x = self.instance_norm2(x)
        x = self.conv3(x)
        x = self.instance_norm3(x)
        return x

class decoder(tf.keras.models.Model):
    def __init__(self):
        super(decoder, self).__init__()
        #we don't need the instance norm and relu in the last layer
        self.conv1 = tf.keras.layers.Conv2DTranspose(64,[3,3],strides=[2,2], padding="SAME",activation='relu')
        self.conv2 = tf.keras.layers.Conv2DTranspose(32,[3,3],strides=[2,2], padding="SAME",activation='relu')
        self.conv3 = tf.keras.layers.Conv2DTranspose(3,[9,9],strides=[1,1], padding="SAME")
        self.instance_norm1 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")
        self.instance_norm2 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")
    def call(self,inputs):
        #the input channel size is 128 (defult)
        x = self.conv1(inputs)
        x = self.instance_norm1(x)
        x = self.conv2(x)
        x = self.instance_norm2(x)
        x = self.conv3(x)
        return x
    
    
#vgg16 to get the style
"""
class vgg16(tf.keras.models.Model):
    def _init_(self):
        super(vgg16, self).__init__()
        #block1
        self.b1conv1 = tf.keras.layers.Conv2D(64,[3,3],strides=[1,1], padding="SAME",activation='relu')
        self.b1conv2 = tf.keras.layers.Conv2D(64,[3,3],strides=[1,1], padding="SAME",activation='relu')
        self.b1maxpool = tf.keras.layers.MaxPool2D([2,2],[2,2])
        #block2
        self.b2conv1 = tf.keras.layers.Conv2D(128,[3,3],strides=[1,1], padding="SAME",activation='relu')
        self.b2conv2 = tf.keras.layers.Conv2D(128,[3,3],strides=[1,1], padding="SAME",activation='relu')
        self.b2maxpool = tf.keras.layers.MaxPool2D([2,2],[2,2])
        #block3
        self.b3conv1 = tf.keras.layers.Conv2D(256,[3,3],strides=[1,1], padding="SAME",activation='relu')
        self.b3conv2 = tf.keras.layers.Conv2D(256,[3,3],strides=[1,1], padding="SAME",activation='relu')
        self.b3conv3 = tf.keras.layers.Conv2D(256,[3,3],strides=[1,1], padding="SAME",activation='relu')
        self.b3maxpool = tf.keras.layers.MaxPool2D([2,2],[2,2])
        #block4
        self.b4conv1 = tf.keras.layers.Conv2D(256,[3,3],strides=[1,1], padding="SAME",activation='relu')
        self.b4conv2 = tf.keras.layers.Conv2D(256,[3,3],strides=[1,1], padding="SAME",activation='relu')
        self.b4conv3 = tf.keras.layers.Conv2D(256,[3,3],strides=[1,1], padding="SAME",activation='relu')
        self.b4maxpool = tf.keras.layers.MaxPool2D([2,2],[2,2])
        #block5
        self.b5conv1 = tf.keras.layers.Conv2D(256,[3,3],strides=[1,1], padding="SAME",activation='relu')
        self.b5conv2 = tf.keras.layers.Conv2D(256,[3,3],strides=[1,1], padding="SAME",activation='relu')
        self.b5conv3 = tf.keras.layers.Conv2D(256,[3,3],strides=[1,1], padding="SAME",activation='relu')
        self.b5maxpool = tf.keras.layers.MaxPool2D([2,2],[2,2])
        
    def call(self,inputs):
        x = self.b1conv1(inputs)
        x = self.b1conv2(x)
        x = self.b1maxpool(x)
        
        x = self.b2conv1(x)
        x = self.b2conv2(x)
        x = self.b2maxpool(x)
        
        x = self.b3conv1(x)
        x = self.b3conv2(x)
        x = self.b3conv3(x)
        x = self.b3maxpool(x)
        
        x = self.b4conv1(x)
        x = self.b4conv2(x)
        x = self.b4conv3(x)
        x = self.b4maxpool(x)
        
        x = self.b5conv1(x)
        x = self.b5conv2(x)
        x = self.b5conv3(x)
        x = self.b5maxpool(x)
        return x
    def call_for_cnn_layer(self,x,layer):
        cnn_layers =[self.b1_conv1,self.b1_conv2,self.b2_conv1,self.b2_conv2,
                     self.b3_conv1,self.b3_conv2,self.b3_conv3,
                     self.b4_conv1,self.b4_conv2,self.b4_conv3,
                     self.b5_conv1,self.b5_conv2,self.b5_conv3
                    ]
        return cnn_layers[layer].call(x)
    
"""       

class StyleBank(tf.keras.models.Model):
    def __init__(self, total_style,total_content):
        super(StyleBank, self).__init__()
        self.total_style = total_style
        self.total_content = total_content
        self.encoder = encoder()
        self.decoder = decoder()
        
        self.style_bank = {k: None for k in range(self.total_style)}  
        for i in self.style_bank:
            print("try to build")
            self.style_bank[i] = [
                    tf.keras.layers.Conv2D(256,[9,9],strides=[1,1], padding="SAME",activation='relu'),
                    tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,
                                                                      beta_initializer="random_uniform",
                                                                      gamma_initializer="random_uniform"),
                    tf.keras.layers.Conv2D(256,[3,3],strides=[2,2], padding="SAME",activation='relu'),
                    tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,
                                                                      beta_initializer="random_uniform",
                                                                      gamma_initializer="random_uniform"),
                    tf.keras.layers.Conv2D(128,[3,3],strides=[2,2], padding="SAME",activation='relu'),                    
                    tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,
                                                                      beta_initializer="random_uniform",
                                                                      gamma_initializer="random_uniform"),
                    ]
        
    def forward_style_n(self, x, style_n):
        y = self.encoder(x)
        for layer in self.selfbank[style_n]:
            y = layer.call(y)
        return y
    
    def apply_style(self,content,style_indexs):
        print(style_indexs)
        print([self.forward_style_n(content,style_n) for style_n in range(style_indexs)])
        
        #y = tf.stack(),axis = 0)
        #y = tf.gather_nd(y, style_indexs)
        #return y
    
    def call(self,input_images,style_id = None):
        if style_id is not None:
            if isinstance(style_id, int):
                style_id = [style_id]
                input_images = tf.convert_to_tensor(input_images, np.float32)
        
        input_images = self.encoder(input_images)
        
        if style_id is not None:
            # print 'using style mode ... ... ...'
            input_images = self.apply_style(self,input_images,style_id)
        
        return self.decoder(input_images)
    
    
    """
        loss function
            #1.auto-ecoder branch :mse (output - input)**2
            #2.stylizing branch: perceptual loss Lk, 
                                 content loss Lc, 
                                 style loss Ls: feature map() gram_matrix(vgg16)
                                                 inputs: , 
                                 variation relularization Ltv
                                 
    """
    
    def initialize_Ls(output_image_features,style_features):
            #only need 2 4 6 9 cnn_layer
            #use the gram_matrix after the vgg_layer output
        style_loss = 0
        for content_grams, style_grams in zip(output_image_features,style_features):
            style_loss += tf.reduce_mean(tf.squre(content_grams- style_grams))
            #style_loss = style_loss * style_loss_param/len(output_image_features)
        return style_loss
        
        #initialize content loss!!!!!
    def initialize_Lc(vgg16,styled_content_result,input_content_image):
        content_loss = tf.reduce_mean(
            tf.square(
                vgg16(styled_content_result) -
                vgg16(input_content_image)
                )
        )
        return content_loss
    
    def initialize_Li(self,x):
        reconstruct_output = self.forward(x)
        encoder_loss = tf.reduce_mean(tf.square(self.tfX - reconstruct_output))
        return encoder_loss
    
    def sample_train_batch(self, batch_size, img_set):
        if img_set == "content":
            return np.random.choice(np.arange(0, self.total_content), size=batch_size, replace=False)

        elif img_set == "style":
            return np.random.choice(np.arange(0, self.total_style), size=batch_size, replace=False)
        else:
            raise ValueError(
                    "Expected the 'img_set' parameter to be either 'content' or 'style', received '%s' instead." % img_set
                )
        
        
                
                   
"""        
                    
                    print("Epoch: %d. Reconstruct loss: %.2f. Style branch loss: %.2f. Style loss: %.2f. "
                          "Content loss: %.2f. Style indices: %s" % (
                            i,
                            self.reconstruct_losses[-1],
                            self.style_branch_losses[-1],
                            self.style_losses[-1],
                            self.content_losses[-1],
                            str(style_indices)
                        )
                    )
"""