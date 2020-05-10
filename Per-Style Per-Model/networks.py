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
        self.instance_norm1 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True )
        self.instance_norm2 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True)
        self.instance_norm3 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True)
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
        self.instance_norm1 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True)
        self.instance_norm2 = tfa.layers.InstanceNormalization(axis=3, center=True, scale=True)
    def call(self,inputs):
        #the input channel size is 128 (defult)
        x = self.conv1(inputs)
        x = self.instance_norm1(x)
        x = self.conv2(x)
        x = self.instance_norm2(x)
        x = self.conv3(x)
        return x    

class StyleBank(tf.keras.models.Model):
    def __init__(self, total_style,total_content):
        super(StyleBank, self).__init__()
        self.total_style = total_style
        self.total_content = total_content
        self.encoder = encoder()
        self.decoder = decoder()
        self.input_content_image_encoded  = None
        
        
        self.style_bank = {str(k): None for k in range(self.total_style)}  
        for i in self.style_bank:
            names = "style {}".format(i)
            self.style_bank[i] = [
                    tf.keras.layers.Conv2D(256,[3,3],strides=[1,1], padding="SAME",activation='relu',name = names+"Conv_1"),
                    tfa.layers.InstanceNormalization(axis=3, center=True, scale=True),
                    tf.keras.layers.Conv2D(256,[3,3],strides=[1,1], padding="SAME",activation='relu',name = names+"Conv_2"),
                    tfa.layers.InstanceNormalization(axis=3, center=True, scale=True),
                    tf.keras.layers.Conv2D(128,[3,3],strides=[1,1], padding="SAME",activation='relu',name = names+"Conv_3"),                    
                    tfa.layers.InstanceNormalization(axis=3, center=True, scale=True)
            ]
            self.style_bank[i][0].build((None,56, 56, 128))
            self.style_bank[i][1].build((None, 56, 56, 256))
            self.style_bank[i][2].build((None, 56, 56, 256))
            self.style_bank[i][3].build((None, 56, 56, 256))
            self.style_bank[i][4].build((None, 56, 56, 256))
            self.style_bank[i][5].build((None, 56, 56, 128))
    def encode(self,inputs):
        return self.encoder(inputs)
    def decode(self,inputs):
        return self.decoder(inputs)
    
    def forward_style_n(self, x, style_n):
        y = self.encoder(x)
        for layer in self.style_bank[str(style_n)]:
            y = layer.call(y)
        y = self.decoder(y)
        return y
    
    def apply_style(self,content,style_indexs):
        style_indices = np.array([
                    [style_n, k] for k, style_n in enumerate(range(len(style_indexs)))
                ])
        y = tf.stack([self.forward_style_n(content,i) for i in style_indexs],axis = 0)  
        y = tf.gather_nd(y, style_indices)
        return y
    
    def call(self,input_images,style_id = None):
        if isinstance(style_id, int):
            style_id = [style_id]
            input_images = tf.convert_to_tensor(input_images, np.float32)
        
        input_images = self.encoder(input_images)
        
        if style_id is not None:
            # print 'using style mode ... ... ...'
            input_images = self.apply_style(input_images,style_id)
        
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
    
    def initialize_Ls(self,output_image_features,style_features):
            #only need 2 4 6 9 cnn_layer
            #use the gram_matrix after the vgg_layer output
        style_loss = 0
        for out_grams, style_grams in zip(output_image_features,style_features):
            style_loss += tf.reduce_mean(tf.math.square(out_grams- style_grams))
            #style_loss = style_loss * style_loss_param/len(output_image_features)
        return style_loss
        
        #initialize content loss!!!!!
    def initialize_Lc(self,output_image_features,input_content_image):
        content_loss = 0
        for out_grams, content_grams in zip(output_image_features,input_content_image):
            content_loss += tf.reduce_mean(
                tf.square(
                    out_grams- content_grams
                )
            )
        return content_loss
    
    def initialize_Li(self,x):
        reconstruct_output = self.encoder(x)
        reconstruct_output = self.decoder(reconstruct_output)
        encoder_loss = tf.reduce_mean(tf.math.square(x - reconstruct_output))
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
        
        
                
                   
