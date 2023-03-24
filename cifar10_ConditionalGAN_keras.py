
###   ------importing libraries --------  ###

from numpy.random import randn,randint
from numpy import zeros,ones
from numpy import asarray
import sys

from keras.layers import Input,Dense,Reshape,Flatten
from keras.layers import Conv2D,Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout,Embedding,Concatenate

from keras.models import Model
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
from keras.datasets.cifar10 import load_data
import matplotlib.pyplot as plt


###  ------- Discriminator and generator -------- ###

def create_discriminator(inshape=(32,32,3),n_classes=10):
    in_label=Input(shape=(1,))
    
    em_label=Embedding(n_classes,50)(in_label)
    num_nodes=inshape[0]*inshape[1]

    em_label=Dense(num_nodes)(em_label)

    em_label=Reshape((inshape[0],inshape[1],1))(em_label)
    
    in_image=Input(shape=inshape)
    mergeInput=Concatenate()([in_image,em_label])
    
    layer=Conv2D(128, (3,3), strides=(2,2), padding='same')(mergeInput)   
    layer=LeakyReLU(alpha=.2)(layer)

    layer=Conv2D(128, (3,3), strides=(2,2), padding='same')(layer)
    layer=LeakyReLU(alpha=.2)(layer)

    layer=Flatten()(layer)
    layer=Dropout(.4)(layer)
    
    out_layer=Dense(1, activation='sigmoid')(layer)
    
    model=Model([in_image,in_label],out_layer)
    
    optim=Adam(lr=.0002,beta_1=.5)
    model.compile(loss='binary_crossentropy',optimizer=optim,metrics=['accuracy'])
    return model

#model testing
discirNet=create_discriminator()
print(discirNet.summary())

def create_generator(latent_dim=100,n_classes=10):
    in_label=Input(shape=(1,))
    
    em_label=Embedding(n_classes,50)(in_label)
    
    num_nodes=8*8
    em_label=Dense(num_nodes)(em_label)
    em_label=Reshape((8,8,1))(em_label)
    
      
    in_lat=Input(shape=(latent_dim,))
    num_nodes_img=128*8*8
    layer=Dense(num_nodes_img)(in_lat)
    layer=LeakyReLU(alpha=.2)(layer)
    layer=Reshape((8,8,128))(layer)
    
    mergeddata=Concatenate()([layer,em_label])
    
    layer=Conv2DTranspose(128, (4,4), strides=(2,2),padding='same')(mergeddata)
    layer=LeakyReLU(alpha=.2)(layer)
    
    layer=Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(layer)
    layer=LeakyReLU(alpha=.2)(layer)
    
    out_layer=Conv2D(3, (8,8), activation='tanh',padding='same')(layer)
    
    model=Model([in_lat,in_label],out_layer)
    return model

generatorNet=create_generator()
print(generatorNet.summary())


def create_GAN(generator,discriminator):
    discriminator.trainable=False
    
    gen_noise,gen_label=generator.input
    gen_output=generator.output
    
    gan_output=discriminator([gen_output,gen_label])
    
    model=Model([gen_noise,gen_label],gan_output)
    
    optim=Adam(lr=.0002,beta_1=.5)
    model.compile(loss='binary_crossentropy',optimizer=optim)
    return model

GAN_MODEL=create_GAN(generatorNet, discirNet)   
print(GAN_MODEL.summary())    

    
def data_loader():
    (trainX,trainY),(testX,testY)=load_data()
    trainX=trainX.astype('float32')
    trainX=(trainX-127.5)/127.5
    return [trainX,trainY]

def choice_real_samples(dataset,n_samples):
    images,labels=dataset
    
    index=randint(0,images.shape[0],n_samples)
    choiced_img,choiced_label=images[index],labels[index]
    
    real_label=ones((n_samples,1))
    return [choiced_img,choiced_label], real_label

def generate_latent_points(latent_dim,n_samples,n_classes=10):
    input=randn(latent_dim*n_samples)
    input=input.reshape(n_samples,latent_dim)
    
    labels=randint(0,n_classes,n_samples)
    return [input,labels]

def generate_fake_images(generator,latent_dim,n_samples):
    input,labels=generate_latent_points(latent_dim, n_samples)
    
    generatedImg=generator.predict([input,labels])
    
    fake_label=zeros((n_samples,1))
    return [generatedImg,labels], fake_label

def model_training(genModel,disModel,GANModel,dataset,latent_dim,numEpoch=100,numBatch=128):
    batch_epoch=int(dataset[0].shape[0]/numBatch)
    half_batch=int(numBatch/2)
    
    for epoch in range(numEpoch):
        for batch in range(batch_epoch):
            [choiced_img,choiced_label], real_label=choice_real_samples(dataset, half_batch)
            
            dis_loss_real,_=disModel.train_on_batch([choiced_img,choiced_label],real_label)
            
            [generatedImg,labels], fake_label=generate_fake_images(genModel, latent_dim, half_batch)
            dis_loss_fake=disModel.train_on_batch([generatedImg,labels], fake_label)            
            
            [input,labels]=generate_latent_points(latent_dim, numBatch)
            
            label_gan=ones((numBatch,1))
            
            gan_loss=GANModel.train_on_batch([input,labels],label_gan)
            msg=('Epoch>%d, Batch%d/%d, d1=%.3f, d2=%.3f g=%.3f max_epoch=%d' %
				(epoch+1, batch+1, batch_epoch, dis_loss_real, dis_loss_fake, gan_loss,numEpoch))
        
            
            sys.stdout.write('\r'+msg)
    genModel.save('cifar_condition_generator.h5')
    disModel.save('cifar_condition_discriminator.h5')
    GANModel.save('cifar_condution_GAN.h5')

            
#training           
discrimiNet=create_discriminator()
generaNet=create_generator()
GAN_model=create_GAN(generaNet, discrimiNet)

dataset=data_loader() 
model_training(generaNet,discrimiNet,GAN_model,dataset,100,20)   
    
    
###  ----------MODEL testing  ------------ ###
model=load_model('cifar_condition_generator.h5')

input,labels=generate_latent_points(100, 10)
labels=asarray(labels)

generatedImg=model.predict([input,labels])

generatedImg=(generatedImg+1)/2
generatedImg=(generatedImg*255).astype(np.uint8)

categories=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
fig,axs=plt.subplots(2,5,figsize=(10,6))
for i,ax in enumerate(axs.flatten()):
    ax.imshow(generatedImg[i])
    ax.text(10,0,categories[labels[i]])
    ax.axis('off')
plt.tight_layout()
plt.show()

labels=data_loader()





