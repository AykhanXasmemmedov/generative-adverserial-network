
### -----------------libraries----------------------###

#numpy
import numpy as np
from numpy import zeros,ones
from numpy.random import randint
from numpy import asarray,load
from numpy import vstack
from numpy import savez_compressed

#keras
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Activation

from keras.models import Model
from keras.models import Input
from keras.models import load_model

from keras.initializers import RandomNormal

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

#tensorflow
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

#visualition
import matplotlib.pyplot as plt

import sys
import os

###  ------------------------------ MODELS ------------------------------- ###

##  -------------DISCRIMINATOR--------------  ##
def create_discriminator(image_shape):
    init=RandomNormal(stddev=.002) # weight initialization

    in_satel_image=Input(shape=image_shape)#(256,256,3)
    in_google_map_img=Input(shape=image_shape)#(256,256,3)
    
    mergedInput=Concatenate()([in_satel_image,in_google_map_img])#(256,256,6)
    
    layer=Conv2D(64,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(mergedInput)
    layer=LeakyReLU(alpha=.2)(layer) #(128,128,64)
    
    layer=Conv2D(128,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(layer)
    layer=BatchNormalization()(layer)
    layer=LeakyReLU(alpha=.2)(layer) #(64,64,128)
    
    layer=Conv2D(256,(4,4),(2,2),padding='same',kernel_initializer=init)(layer)
    layer=BatchNormalization()(layer)
    layer=LeakyReLU(alpha=.2)(layer)#(32,32,256)
    
    layer=Conv2D(512,(4,4),(2,2),padding='same',kernel_initializer=init)(layer)
    layer=BatchNormalization()(layer)
    layer=LeakyReLU(alpha=.2)(layer)#(16,16,512)
    
    layer=Conv2D(512,(4,4),padding='same',kernel_initializer=init)(layer)
    layer=BatchNormalization()(layer)
    layer=LeakyReLU(alpha=.2)(layer)#(16,16,512)
    
    layer=Conv2D(1,(4,4),padding='same',kernel_initializer=init)(layer)#(16,16,1)
    
    output=Activation('sigmoid')(layer)
     
    model=Model([in_satel_image,in_google_map_img],output)
    
    optim=Adam(lr=.0002,beta_1=.5)
    model.compile(loss='binary_crossentropy',optimizer=optim,loss_weights=[.5])
        
    return model
#model testing------------
discirmodel=create_discriminator((256,256,3))
print(discirmodel.summary())   
plot_model(discirmodel,'discirmodel.png',show_shapes=True) 



### ------------GENERATOR MODEL----------------------###

##  ------encoder--------##
def encoder_block(layer,filters,batchNorm=True):
    init=RandomNormal(stddev=.02)
    
    layer=Conv2D(filters,(4,4),strides=(2,2),padding='same',\
                 kernel_initializer=init)(layer)
    
    if batchNorm:
        layer=BatchNormalization()(layer,training=True)
        
    layer=LeakyReLU(alpha=.2)(layer)
    
    return layer

## ------decoder---------##
def decoder_block(layer,skip,filters,dropout=True):
        init=RandomNormal(stddev=.02)
        
        layer=Conv2DTranspose(filters, (4,4), strides=(2,2),padding='same',\
                              kernel_initializer=init)(layer)
            
        layer=BatchNormalization()(layer,training=True)
        
        if dropout:
            layer=Dropout(.5)(layer,training=True)
            
        layer=Concatenate()([layer,skip])
        
        layer=Activation('relu')(layer)
        
        return layer

## ------generator model-------##
def create_generator(image_shape):
    init=RandomNormal(stddev=.02)
    
    input_img=Input(shape=image_shape)  #(256,256,3)
    
    ###-------------------------ENCODER--------------------------------------###

    encoder_1=encoder_block(input_img, 64, False)#(128,128,64)
    
    encoder_2=encoder_block(encoder_1,128) #(64,64,128)
    
    encoder_3=encoder_block(encoder_2, 256) #(32,32,256)
    
    encoder_4=encoder_block(encoder_3,512) #(16,16,512)
    encoder_5=encoder_block(encoder_4,512) #(8,8,512)
    encoder_6=encoder_block(encoder_5,512) #(4,4,512)
    encoder_7=encoder_block(encoder_6,512) #(2,2,512)

    
    ### -------------------------BOTTLENECK --------------------------------###
    bottleneck=Conv2D(512,(4,4),strides=(2,2),\
                      padding='same',kernel_initializer=init)(encoder_7)    

    bottleneck=Activation('relu')(bottleneck) #(1,1,512)
    
    ### ---------------------------DECODER --------------------------------###
    
    decoder_1=decoder_block(bottleneck, encoder_7, 512)
    decoder_2=decoder_block(decoder_1, encoder_6, 512)
    decoder_3=decoder_block(decoder_2, encoder_5, 512)
    
    decoder_4=decoder_block(decoder_3, encoder_4, 512, dropout=False)
    decoder_5=decoder_block(decoder_4, encoder_3, 256, dropout=False)
    decoder_6=decoder_block(decoder_5, encoder_2, 128, dropout=False)
    decoder_7=decoder_block(decoder_6, encoder_1, 64, dropout=False)
    
    output=Conv2DTranspose(3, (4,4), strides=(2,2), padding='same',\
                           kernel_initializer=init)(decoder_7)
    
    output=Activation('tanh')(output)
    
    model=Model(input_img,output)
    
    return model

## model tesing------
genertorModel=create_generator((256,256,3))
print(genertorModel.summary())
plot_model(genertorModel, to_file='genertorModel.png', show_shapes=True)




## ------------------------------GAN MODEL ---------------------------------------##
def create_gan(generaModel,discriminModel,image_shape):
    for layer in discriminModel.layers:
        if not isinstance(layer,BatchNormalization):
            layer.trainable=False
            
            
    input_img=Input(shape=image_shape)
    
    gene_out=generaModel(input_img)
    dis_out=discriminModel([input_img,gene_out])
    
    model=Model(input_img,[dis_out,gene_out])
    optim=Adam(lr=.0002,beta_1=.5)
    
    model.compile(loss=['binary_crossentropy','mae'],optimizer=optim,loss_weights=[1,100])
    return model

gan_model=create_gan(genertorModel, discirmodel, (256,256,3))
print(gan_model.summary())
plot_model(gan_model,'gan_model.png',show_shapes=True)


### ---------------------------dataset--------------------------------###
def create_real_dataset(dataset,num_sample,patchShape):
    train_img_satel,train_img_google=dataset
    
    index=randint(0,train_img_satel.shape[0],num_sample)
    
    selected_img_satel,selected_img_google=train_img_satel[index],train_img_google[index]

    label_value=ones((num_sample,patchShape,patchShape,1))    
    return [selected_img_satel,selected_img_google],label_value


def create_fake_dataset(genModel,samples,pathc_shape):
    google_map=genModel(samples)
    
    label=zeros((len(google_map),pathc_shape,pathc_shape,1))

    return google_map,label


### ---------------- function for model training ------------------###
def model_training(discriModel,genModel,GAN_model,dataset,numepoch=100,n_batch=20):
    numpatch=discriModel.output_shape[1]
    
    satellit_img,google_map=dataset
    
    batch_epoch=int(len(satellit_img)/n_batch)
    
    
    for epoch in range(numepoch):
        for batch in range(batch_epoch):
            
            [real_img_satel,real_img_google],real_label=create_real_dataset(dataset, n_batch, numpatch)
            
            fake_google_map,fake_label=create_fake_dataset(genModel, real_img_satel, numpatch)
    
    
            dis_loss_real=discriModel.train_on_batch([real_img_satel,real_img_google],real_label)
            dis_loss_fake=discriModel.train_on_batch([real_img_satel,fake_google_map],fake_label)
            
            gan_loss,_,_=GAN_model.train_on_batch(real_img_satel,[real_label,real_img_google])
            
            msg=('Epoch>%d, Batch%d/%d, d1=%.3f, d2=%.3f g=%.3f max_epoch=%d' %
				(epoch+1, batch+1, batch_epoch, dis_loss_real, dis_loss_fake, gan_loss,numepoch))
                    
            sys.stdout.write('\r'+msg)  
            
    genModel.save('generator_model_150Epochs.h5')
    discriModel.save('deiscriminator_model_150Epochs.h5')
    GAN_model.save('GAN_model_150Epochs.h5')
            

# data loading to memory and training

def load_images(images_path,size=(256,512)):

    
    satellite_list,google_map_list=list(),list()

    for imageName in os.listdir(images_path):
        image=load_img(os.path.join(images_path,imageName),target_size=size)
        
        image=img_to_array(image)
        
        sat_img,google_img=image[:,:256],image[:,256:]
        satellite_list.append(sat_img)
        google_map_list.append(google_img)
        
    return asarray(satellite_list),asarray(google_map_list)
        

def data_proprocess(satellit,google_map):
    satellit=(satellit-127.5)/127.5
    google_map=(google_map-127.5)/127.5
    
    return satellit,google_map

        
path=os.getcwd()
images_path=os.path.join(path,'maps','val')
satellite_imgs,google_map_imgs=load_images(images_path)

for i in range(5):
    index=randint(0,satellite_imgs.shape[0])
    fig,ax=plt.subplots(1,2,figsize=(10,6))
    ax[0].imshow(satellite_imgs[index].astype('uint8'))
    ax[1].imshow(google_map_imgs[index].astype('uint8'))


image_shape=satellite_imgs.shape[1:]

discriminator_model=create_discriminator(image_shape)
generator_model=create_generator(image_shape)
gan_model=create_gan(generator_model, discriminator_model, (256,256,3))


discriminator_model=load_model('deiscriminator_model_60Epochs.h5')
generator_model=load_model('generator_model_60Epochs.h5')
gan_model=load_model('GAN_model_60Epochs.h5')

satellite_imgs,google_map_imgs=data_proprocess(satellite_imgs, google_map_imgs)
dataset=[satellite_imgs,google_map_imgs]
model_training(discriminator_model, generator_model, gan_model, dataset,70,1)


###  ----------------GENERATOR MODEL TESTING ----------------------- ###

model=load_model('generator_model_60Epochs.h5')

path=os.getcwd()
images_path=os.path.join(path,'maps','train')
test_satellite_img,test_google_map=load_images(images_path)
test_satellite_img,test_google_map=data_proprocess(test_satellite_img, test_google_map)

sampleNum=1
index=randint(0,test_google_map.shape[0],sampleNum)
selected_satellite,selected_google_map=test_satellite_img[index],test_google_map[index]

# model predict
generated_map=model.predict(selected_satellite)

# output visualition
selected_satellite=(((selected_satellite+1)/2)*255).astype('uint8')
selected_google_map=(((selected_google_map+1)/2)*255).astype('uint8')
generated_map=(((generated_map+1)/2)*255).astype('uint8')

for i in range(sampleNum):
    fig,ax=plt.subplots(1,3,figsize=(10,6))
    ax[0].imshow(selected_satellite[i].astype('uint8'))
    ax[1].imshow((selected_google_map[i]).astype('uint8'))
    ax[1].set_title('orginal')
    ax[2].imshow(generated_map[i].astype('uint8'))
    ax[2].set_title('generated')
    
plt.show()


