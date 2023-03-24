### ------------Libraries-------------------###
import numpy as np
import matplotlib.pyplot as plt
import sys

from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2DTranspose

from keras.models import Sequential
from keras.models import load_model

from tensorflow.keras.optimizers import Adam
from keras.datasets.cifar10 import load_data




### --------------discriminator and generator models------------------  ###


def discriminator(in_shape=(32,32,3)):
    model=Sequential() 

    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same',input_shape=in_shape))
    model.add(LeakyReLU(alpha=.2))

    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dropout(.4))
    model.add(Dense(1,activation='sigmoid'))

    optim=Adam(lr=.00002, beta_1=.5)
    model.compile(loss='binary_crossentropy', optimizer=optim,metrics=['accuracy'])
    return model

disNet=discriminator()
print(disNet.summary())


def generator(latent_dim=100):
    model=Sequential()

    n_nodes=256*4*4
    model.add(Dense( n_nodes,input_dim=latent_dim))
    model.add(LeakyReLU(alpha=.2))
    model.add(Reshape((4,4,256)))

    model.add(Conv2DTranspose(128,(4,4), strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=.2)) #8x8x128

    model.add(Conv2DTranspose(32,(4,4),strides=(2,2),padding='same'))
    model.add(LeakyReLU(alpha=.2)) #16x16x32

    model.add(Conv2DTranspose(4, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=.2))

    model.add(Conv2D(3,(4,4),strides=(1,1),activation='tanh',padding='same'))
    return model

genNet=generator()
print(genNet.summary())



def GAN_MODEL(generator,discriminator):
    discriminator.trainable=False

    model=Sequential()
    model.add(generator)
    model.add(discriminator)

    optim=Adam(lr=.0002,beta_1=.5)
    model.compile(loss='binary_crossentropy',optimizer=optim)
    return model

def load_data_samples():
    (trainX,trainY),(testX,testY)=load_data()
    trainX=trainX.astype('float32')
    trainX=(trainX-127.5)/127.5
    return trainX

def random_choice_samples(dataset,n_samples):
    index=np.random.randint(0,dataset.shape[0],n_samples)
    data=dataset[index]
    label=np.ones((n_samples,1))
    return data,label

def generate_latent_points(latent_dim,n_samples):
    sample=np.random.randn(latent_dim*n_samples)
    sample=sample.reshape(n_samples,latent_dim)
    return sample
    
def generate_fake_images(generator,latent_dim,n_samples):
    sample=generate_latent_points(latent_dim,n_samples)
    
    image=generator.predict(sample)
    label=np.zeros((n_samples,1))
    return image,label
    


def model_training(genModel,disModel,GAN_model,dataset,latent_dim=100,numepoch=100,nbatch=128):
    
    batch_epoch=int(dataset.shape[0]/nbatch)
    half_batch=int(nbatch/2)
    
    for epoch in range(numepoch):
        for bat in range(batch_epoch):
            # discriminator
            data_real,label_real=random_choice_samples(dataset=dataset,n_samples=half_batch)
            dis_loss_real,accuracy=disModel.train_on_batch(data_real,label_real)
            
            data_fake,label_fake=generate_fake_images(generator=genModel, latent_dim=latent_dim,n_samples=half_batch)
            dis_loss_fake,accuracy=disModel.train_on_batch(data_fake,label_fake)
            
            
            # generator
            genSample=generate_latent_points(latent_dim=latent_dim, n_samples=nbatch)
            labelGene=np.ones((nbatch,1))
            
            gen_loss=GAN_model.train_on_batch(genSample,labelGene)
            
            msg=('Epoch>%d, Batch %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(epoch+1, bat+1, batch_epoch, dis_loss_real, dis_loss_fake, gen_loss))
            sys.stdout.write('\r'+msg)
            
    genModel.save('cifar_generator.h5')
    disModel.save('cifar_discrimnator.h5')
    GAN_model.save('cifar_GAN.h5')
      
discriminatorNet=discriminator()
generatorNet=generator()
GAN_model=GAN_MODEL(generatorNet,discriminatorNet)

dataset=load_data_samples()

model_training(generatorNet,discriminatorNet,GAN_model,dataset,100,250)



###  --------------generator model testing----------------  ###

model=load_model('cifar_generator.h5')
latentpoint=generate_latent_points(100, 10)

generatedImage=model.predict(latentpoint)
generatedImage=((generatedImage+1)/2 )
generatedImage=(generatedImage*255).astype(np.uint8)

fig,axs=plt.subplots(2,5,figsize=(10,6))
for i,ax in enumerate(axs.flatten()):
    
    ax.imshow(generatedImage[i,:,:,:])
    ax.axis('off')

plt.tight_layout()
plt.show()
