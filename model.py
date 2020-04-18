# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:13:51 2020

@author: 潘慧杰
"""

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
class vgg16(tf.keras.models.Model):
    def _init_(self,trainable = True):
        super(vgg16, self).__init__()
        self.trainable = trainable
        #block1
        self.b1_conv1 = tf.keras.layers.Conv2D(64,[3,3],strides=[1,1], padding="SAME",activation='relu')
        self.b1_conv2 = tf.keras.layers.Conv2D(64,[3,3],strides=[1,1], padding="SAME",activation='relu')
        self.b1_maxpool = tf.keras.layers.MaxPool2D([2,2],[2,2])
        #block2
        self.b2_conv1 = tf.keras.layers.Conv2D(128,[3,3],strides=[1,1], padding="SAME",activation='relu')
        self.b2_conv2 = tf.keras.layers.Conv2D(128,[3,3],strides=[1,1], padding="SAME",activation='relu')
        self.b2_maxpool = tf.keras.layers.MaxPool2D([2,2],[2,2])
        #block3
        self.b3_conv1 = tf.keras.layers.Conv2D(256,[3,3],strides=[1,1,1,1], padding="SAME",activation='relu')
        self.b3_conv2 = tf.keras.layers.Conv2D(256,[3,3],strides=[1,1,1,1], padding="SAME",activation='relu')
        self.b3_conv3 = tf.keras.layers.Conv2D(256,[3,3],strides=[1,1,1,1], padding="SAME",activation='relu')
        self.b3_maxpool = tf.keras.layers.MaxPool2D([2,2],[2,2])
        #block4
        self.b4_conv1 = tf.keras.layers.Conv2D(256,[3,3],strides=[1,1,1,1], padding="SAME",activation='relu')
        self.b4_conv2 = tf.keras.layers.Conv2D(256,[3,3],strides=[1,1,1,1], padding="SAME",activation='relu')
        self.b4_conv3 = tf.keras.layers.Conv2D(256,[3,3],strides=[1,1,1,1], padding="SAME",activation='relu')
        self.b4_maxpool = tf.keras.layers.MaxPool2D([2,2],[2,2])
        #block5
        self.b5_conv1 = tf.keras.layers.Conv2D(256,[3,3],strides=[1,1,1,1], padding="SAME",activation='relu')
        self.b5_conv2 = tf.keras.layers.Conv2D(256,[3,3],strides=[1,1,1,1], padding="SAME",activation='relu')
        self.b5_conv3 = tf.keras.layers.Conv2D(256,[3,3],strides=[1,1,1,1], padding="SAME",activation='relu')
        self.b5_maxpool = tf.keras.layers.MaxPool2D([2,2],[2,2])
    def call(self,inputs,training):
        x = self.b1_conv1(inputs)
        x = self.b1_conv2(x)
        x = self.b1_maxpool(x)
        
        x = self.b2_conv1(x)
        x = self.b2_conv2(x)
        x = self.b2_maxpool(x)
        
        x = self.b3_conv1(x)
        x = self.b3_conv2(x)
        x = self.b3_conv3(x)
        x = self.b3_maxpool(x)
        
        x = self.b4_conv1(x)
        x = self.b4_conv2(x)
        x = self.b4_conv3(x)
        x = self.b4_maxpool(x)
        
        x = self.b5_conv1(x)
        x = self.b5_conv2(x)
        x = self.b5_conv3(x)
        x = self.b5_maxpool(x)
    def load_wight(self,weight_path):
        pass
    def call_for_cnn_layer(self,x,layer):
        cnn_layers =[self.b1_conv1,self.b1_conv2,self.b2_conv1,self.b2_conv2,
                     self.b3_conv1,self.b3_conv2,self.b3_conv3,
                     self.b4_conv1,self.b4_conv2,self.b4_conv3,
                     self.b5_conv1,self.b5_conv2,self.b5_conv3
                    ]
        return cnn_layers[layer].call(x)
        

        
#stylebank       
class StyleBank(tf.keras.models.Model):
    def __init__(self, img_shape, content_shape, style_imgs_path, content_imgs_path, style_loss_param, content_layer_n):
        super(StyleBank, self).__init__()
        # only input image shape matter
        self.img_shape = img_shape
        self.content_shape = content_shape
        # style image
        
        
        self.style_loss_param = style_loss_param
        self.content_layer_n = content_layer_n
        self.optimizer = None   #get form fit
        self.first_fit = True

        # style bank
        self.style_bank = {k: None for k in range(self.n_styles)}

        # operations
        self.reconstruct_op = None
        self.encoder_loss = None
        self.encoder_train_op = None
        self.style_loss = None
        self.content_loss = None
        self.style_branch_train_op = None

        # other attributes
        self.reconstruct_losses = None
        self.style_branch_losses = None
        self.style_losses = None
        self.content_losses = None
        
        self.encoder = encoder()
        self.decoder = decoder()
        self.vgg16 = vgg16()
        
    def init_style_bank(self):
        print(self.n_styles)
        for i in self.style_bank:
            self.style_bank[i] = [
                    tf.keras.layers.Conv2D(256,[9,9],strides=[1,1], padding="SAME",activation='relu'),
                    tf.keras.layers.Conv2D(256,[3,3],strides=[2,2], padding="SAME",activation='relu'),
                    tf.keras.layers.Conv2D(128,[3,3],strides=[2,2], padding="SAME",activation='relu'),
                    tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,
                                                                      beta_initializer="random_uniform",
                                                                      gamma_initializer="random_uniform"),
                    tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,
                                                                      beta_initializer="random_uniform",
                                                                      gamma_initializer="random_uniform"),
                    tfa.layers.InstanceNormalization(axis=3, center=True, scale=True,
                                                                      beta_initializer="random_uniform",
                                                                      gamma_initializer="random_uniform"),
                    ]
    def forward_style_n(self, x, style_n):
        y = self.encoder(x)
        for layer in self.selfbank[style_n]:
            y = layer.call(y)
        return y
    def apply_style(self,content,style_index):
        y = tf.stack([self.forward_style_n(content,style_n) for style_n in range(self.n_styles)],axis = 0)
        y = tf.gather_nd(y, style_index)
        return y
        
        
        """
        loss function
            #1.auto-ecoder branch :mse (output - input)**2
            #2.stylizing branch: perceptual loss Lk, 
                                 content loss Lc, 
                                 style loss Ls: feature map() gram_matrix(vgg16)
                                                 inputs: , 
                                 variation relularization Ltv
                                 
        """
        #sub-fuction first
    
        
        
        
        #initialize style loss
    def initialize_Ls(self,style_content,style):
            #only need 2 4 6 9 cnn_layer
            #use the gram_matrix after the vgg_layer output
        styled_content_outputs = [
                tf.map_fn(gram_matrix, self.vgg16.call_for_cnn_layer(style_content,i)) for i in[2, 4, 6, 9]
            ]
        style_outputs = [
                tf.map_fn(gram_matrix, self.vgg16.call_for_cnn_layer(style,i)) for i in [2, 4, 6, 9]
            ]
        self.style_loss = 0
        for content_grams, style_grams in zip(styled_content_outputs,style_outputs):
            self.style_loss += tf.reduce_mean(tf.squre(content_grams- style_grams))
            self.style_loss = self.style_loss * self.style_loss_param/len(styled_content_outputs)
        
        #initialize content loss!!!!!
    def initialize_Lc(self,styled_content_result,input_content_image):
        self.content_loss = tf.reduce_mean(
            tf.square(
                self.vgg16.call_for_cnn_layer(styled_content_result, self.content_layer_n) -
                self.vgg16.call_for_cnn_layer(input_content_image, self.content_layer_n)
                )
        )
            
    def initialize_Lk(self,x):
        reconstruct_output = self.reconstruct(x)
        self.encoder_loss = tf.reduce_mean(tf.square(self.tfX - reconstruct_output))
            
        
        
        
        
        #setting the environment
    def reconstruct(self,x):
        y = self.encoder.call(x)
        y = self.decoder.call(y)
        return y
        
    def initialize_style_branch(self, c, s, style_indices):
            # apply the appropriate styles to the content images
        self.style_content_imgs_op = self.apply_styles(c, style_indices)  # self.forward_style_n(c, 0)

            # initialize the style loss
        self.initialize_style_loss(self.style_content_imgs_op, s)

            # initialize the content loss
        self.initialize_content_loss(self.style_content_imgs_op, c)

            # define the style branch loss
        self.style_branch_loss = self.style_loss + self.content_loss
    
    
    #return a rundom branch of content and style
    
    def sample_train_batch(self, batch_size, img_set):
        if img_set == "content":
            return np.random.choice(np.arange(0, self.n_content_imgs), size=batch_size, replace=False)

        elif img_set == "style":
            return np.random.choice(np.arange(0, self.n_styles), size=batch_size, replace=False)
        else:
            raise ValueError(
                    "Expected the 'img_set' parameter to be either 'content' or 'style', received '%s' instead." % img_set
                )
        
        
        
        
        
    def fit(self,n_epochs, n_steps, batch_size, optimizer=None, print_step=20, resume_training=True):
        if self.first_fit or not resume_training:         
            self.optimizer = optimizer if optimizer is not None else tf.optimizers.Adam()
            #loss value
            self.reconstruct_losses = []
            self.style_branch_losses = []
            self.content_losses = []
            self.style_losses = []
            
            self.reconstruct_output = self.reconstruct(self.tfX)
            self.encoder_loss = tf.reduce_mean(tf.square(self.tfX - self.reconstruct_op))
            self.encoder_train_op = self.optimizer.minimize(self.encoder_loss)
            
            self.initialize_style_branch(self.tfX, self.tfS, self.tfStyleIndices)
            # store the w and b into a list
            style_vars = []
            for style_n in self.style_bank:
                style_vars += [
                        layer.kernel for layer in self.style_bank[style_n] if isinstance(layer, tf.keras.layers.Conv2D)
                        ]

                style_vars += [
                        layer.bias for layer in self.style_bank[style_n] if isinstance(layer, tf.keras.layers.Conv2D)
                        ]
            
            self.first_fit = False
                
                
        #preperation
            
    
            
        #train
        for i in range(n_epochs):
            for j in range(n_steps):
                c_i = self.sample_train_batch(batch_size, "content")
                s_i = self.sample_train_batch(batch_size, "style")
                #two list to hold the branch index
                c_batch = content_images[c_i]
                s_batch = style_images[s_i]
                    
                    
                style_indices = np.array([
                    [style_n, k] for k, style_n in enumerate(s_i)
                ])
                
                    
                self.session.run(
                    self.style_branch_train_op,
                    feed_dict={self.tfX: c_batch, self.tfS: s_batch, self.tfStyleIndices: style_indices}
                )
                    
                c_i = self.sample_train_batch(batch_size, "content")
                c_batch = content_images[c_i]
                self.session.run(
                    self.encoder_train_op,
                    feed_dict={self.tfX: c_batch}
                )
                
                if i > 0 and i % print_step == 0:
                    self.reconstruct_losses.append(
                        self.session.run(
                            self.encoder_loss,
                            feed_dict={self.tfX: c_batch}
                        )
                    )
                    
                    
                    self.style_branch_losses.append(
                        self.session.run(
                            self.style_branch_loss,
                            feed_dict={self.tfX: c_batch, self.tfS: s_batch, self.tfStyleIndices: [[0, 0]]}
                        )
                    )
                    
                    self.content_losses.append(
                        self.session.run(
                            self.content_loss,
                            feed_dict={self.tfX: c_batch, self.tfS: s_batch, self.tfStyleIndices: [[0, 0]]}
                        )
                    )
                    
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