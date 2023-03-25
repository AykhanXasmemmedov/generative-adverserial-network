###-----satellite image to maps translation using pix2pix GAN--------###

### --------------------libraries--------------------###
##-----torch------##
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import randn

##---torchvision---##
import torchvision
from torchvision import models,transforms,datasets

#----torchsummary
from torchsummary import summary

#-----numpy
import numpy as np
from numpy.random import randint

#-----map=tplotlib
import matplotlib.pyplot as plt

#-----additional libraries
import os
import sys


### -------------GENERATOR MODEL -----------------###
class create_generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        #channels=[3,64,128,256,512,512,512,512]
        ### bias don't need in large netowrks
        ### being inplace True in activation function create problem in some cases 
        
        ###---------ENCODER----------###
        self.encoder_1=nn.Conv2d(3,64,4,2,1,bias=False)
        
        self.encoder_2=nn.Sequential(
                nn.LeakyReLU(.2,False),
                nn.Conv2d(64,128,4,2,1,bias=False),
                nn.BatchNorm2d(128)
        )
        
        self.encoder_3=nn.Sequential(
                nn.LeakyReLU(.2,False),
                nn.Conv2d(128,256,4,2,1,bias=False),
                nn.BatchNorm2d(256)  
        )
        
        self.encoder_4=nn.Sequential(
                nn.LeakyReLU(.2,False),
                nn.Conv2d(256,512,4,2,1,bias=False),
                nn.BatchNorm2d(512)
        )
        
        self.encoder_5=nn.Sequential(
                nn.LeakyReLU(.2,False),
                nn.Conv2d(512,512,4,2,1,bias=False),
                nn.BatchNorm2d(512) 
        )
        
        self.encoder_6=nn.Sequential(
                nn.LeakyReLU(.2,False),
                nn.Conv2d(512,512,4,2,1,bias=False),
                nn.BatchNorm2d(512) 
        )
        
        self.encoder_7=nn.Sequential(
                nn.LeakyReLU(.2,False),
                nn.Conv2d(512,512,4,2,1,bias=False),
                nn.BatchNorm2d(512)               
        )
        
        ### ----------BOTTLE_NECK--------------###
        
        self.bottleneck=nn.Sequential(
                nn.LeakyReLU(.2,False),
                nn.Conv2d(512,512,4,2,1,bias=False)
                
        )
                
        ### ------------DECODER--------------###
        
        self.decoder_1=nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512,512,4,2,1,bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout(.5)
        )
        
        self.decoder_2=nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024,512,4,2,1,bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout(.5)
        )
        
        self.decoder_3=nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024,512,4,2,1,bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout(.5)
        )
        
        self.decoder_4=nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024,512,4,2,1,bias=False),
            nn.BatchNorm2d(512),
        )
        
        self.decoder_5=nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
        )
        
        self.decoder_6=nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
        )
        
        self.decoder_7=nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256,64,4,2,1,bias=False),
            nn.BatchNorm2d(64),
        )
        
        self.decoder_8=nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128,3,4,2,1,bias=False),
            nn.Tanh()
        )
           
    def forward(self,x):
        ##----------encoder------------##
        enc_1=self.encoder_1(x)
        enc_2=self.encoder_2(enc_1)
        enc_3=self.encoder_3(enc_2)
        enc_4=self.encoder_4(enc_3)
        enc_5=self.encoder_5(enc_4)
        enc_6=self.encoder_6(enc_5)
        enc_7=self.encoder_7(enc_6)
        
        # --------bottle neck---------##
        bottle_neck=self.bottleneck(enc_7)
        
        ## -------decoder ------------##
        
        dec_1= torch.cat([self.decoder_1(bottle_neck),enc_7],dim=1)
        dec_2=torch.cat([self.decoder_2(dec_1),enc_6],dim=1)
        dec_3=torch.cat([self.decoder_3(dec_2),enc_5],dim=1)
        dec_4=torch.cat([self.decoder_4(dec_3),enc_4],dim=1)
        dec_5=torch.cat([self.decoder_5(dec_4),enc_3],dim=1)
        dec_6=torch.cat([self.decoder_6(dec_5),enc_2],dim=1)
        dec_7=torch.cat([self.decoder_7(dec_6),enc_1],dim=1)
        
        output=self.decoder_8(dec_7)
        
        return output
        
## model checking      
generator=create_generator() 
data=randn((1,3,256,256))       
print(generator(data))


class create_discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1=nn.Sequential(
                nn.Conv2d(6,64,4,2,1,bias=False),
                nn.LeakyReLU(.2)   
                
            )
        
        self.layer_2=nn.Sequential(
                nn.Conv2d(64,128,4,2,1,bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(.2)   
                
            )
        
        self.layer_3=nn.Sequential(
                nn.Conv2d(128,256,4,2,1,bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(.2)
            
            )
        
        self.layer_4=nn.Sequential(
                nn.Conv2d(256,512,4,2,1,bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(.2)
                
            )
        
        self.layer_5=nn.Sequential(
                nn.Conv2d(512,512,4,1,1,bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(.2)
            
            )
        
        self.layer_6=nn.Sequential(
                nn.Conv2d(512,1,4,1,1,bias=False),
                nn.Sigmoid()
            
            )

    def forward(self,x):
        x=self.layer_1(x)
        x=self.layer_2(x)
        x=self.layer_3(x)
        x=self.layer_4(x)
        x=self.layer_5(x)
        x=self.layer_6(x)
        
        return x
## model checking
discriminator=create_discriminator()
data=randn((1,6,256,256))
print(discriminator(data))


def weights_init(m):
    name=m.__class__.__name__
    
    if (name.find('Conv')>-1):
        nn.init.normal_(m.weight.data,.0,.02)
    elif(name.find('BatchNorm')>-1):
        nn.init.normal_(m.weight.data,1.0,.02)
        nn.init.constant_(m.bias.data,.0)
        


####////////////--data import --////////////////////////////////////////////////////////////////////////////

path=os.getcwd()
data_dir = "maps_torch"

data_transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.CenterCrop((256, 512)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset_train = datasets.ImageFolder(root=os.path.join(path,data_dir, "train"), transform=data_transform)
dataset_val = datasets.ImageFolder(root=os.path.join(path,data_dir, "val"), transform=data_transform)

dataloader_train=DataLoader(dataset_train,batch_size=1,shuffle=True)
dataloader_val=DataLoader(dataset_val,batch_size=len(dataset_val),shuffle=True)

print(f'data loader train len:{len(dataloader_train)}')
print(f'data loader val len:{len(dataloader_val)}')


def data_visualtion(image,samplenum):
    image=image.permute(0,2,3,1)
    
    index=randint(0,len(image),samplenum)
    
    for i in range(samplenum):
        img=image[index[i]].detach()
        
        img=np.asarray(img)
        img=(((img+1)/2)*255).astype('uint8')
        plt.imshow(img)
        plt.show()

images,_=next(iter(dataloader_train))
data_visualtion(images,1)


###//////////////////////////////////////////////////////////////////////////////////////###
             ### training process ###
             
def model_training(generator,discriminator,dataset,numepoch):
    
    generator.to(device)
    discriminator.to(device)
    
    optimizerGene=optim.Adam(generator.parameters(),lr=.0002,betas=(.5,.99))
    optimizerDis=optim.Adam(discriminator.parameters(),lr=.0002,betas=(.5,.99))

    loss_function_dis=nn.BCELoss()
    loss_function_genL1=nn.L1Loss()
    l1_loss_weight=100
    
    for epoch in range(numepoch):
        batch=0
        
        for data,_ in dataset:
            
            ####------------------discrimintor----------------------------####
            
            satellite=data[:,:,:,:256].to(device)
            google_map=data[:,:,:,256:].to(device)
            
            discriminator.zero_grad()
            
            ###/////////////////////////////////////////////////////////////////////////////
            real_data=torch.cat([satellite,google_map],dim=1)
            real_out=discriminator(real_data)
            real_labels=torch.ones(size=real_out.shape,dtype=torch.float,device=device)
            
            loss_dis_real=.5*loss_function_dis(real_out,real_labels)
            loss_dis_real.backward()
            
            ###//////////////////////////////////////////////////////////////////////////////
            generated_data=generator(satellite)
            fake_data=torch.cat([satellite,generated_data],dim=1)
            fake_out=discriminator(fake_data)
            fake_labels=torch.zeros(size=fake_out.shape,dtype=torch.float,device=device)
            
            loss_dis_fake=.5*loss_function_dis(fake_out,fake_labels)
            loss_dis_fake.backward()
            
            optimizerDis.step()
            
            
            ###------------------- generator -----------------------------####
            
            generator.zero_grad()
            
            generatedImg=generator(satellite)
            
            gene_data=torch.cat([satellite,generatedImg],dim=1)
            output=discriminator(gene_data)
            gene_labels=torch.ones(size=output.shape,dtype=torch.float,device=device)
            
            BinCrossLoss=loss_function_dis(output,gene_labels)
            L1_loss=loss_function_genL1(generatedImg,google_map)
            
            lossGene=BinCrossLoss+l1_loss_weight*L1_loss
            lossGene.backward()
            optimizerGene.step()
            
            ###////////////////// message ////////////////////////////////////////////////
            batch=batch+1
            
            msg=(f'epoch:{epoch+1},-- batch :{batch}') 
            sys.stdout.write('\r'+msg)  
            

    torch.save(generator,'torch_generator.pth')
    torch.save(discriminator,'torch_discriminator.pth')

    return generator,discriminator

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

GeneratorModel=create_generator()
Discriminator=create_discriminator()

GeneratorModel.apply(weights_init)
Discriminator.apply(weights_init)

GeneratorModel=torch.load('torch_generator.pth')
Discriminator=torch.load('torch_discriminator.pth')

GeneratorModel,Discriminator=model_training(GeneratorModel, Discriminator, dataloader_train, 5)




####////////////////////////////////////////////////////////////////////////////////////////////////////////////
                     #### GENERATOR MODEL TESTING ####
def data_visualtion_3part(generated,satellite,maps,samplenum):
    generated=generated.permute(0,2,3,1)
    satellite=satellite.permute(0,2,3,1)
    maps=maps.permute(0,2,3,1)
    
    index=randint(0,len(generated),samplenum)
    
    for i in range(samplenum):
        gene=generated[index[i]].detach()
        satel=satellite[index[i]].detach()
        mapimg=maps[index[i]].detach()
        
        gene=np.asarray(gene)
        gene=(((gene+1)/2)*255).astype('uint8')
        
        satel=np.asarray(satel)
        satel=(((satel+1)/2)*255).astype('uint8')
        
        mapimg=np.asarray(mapimg)
        mapimg=(((mapimg+1)/2)*255).astype('uint8')
        
        fig,ax=plt.subplots(1,3,figsize=(10,6))
        ax[0].imshow(satel)
        ax[1].imshow(mapimg)
        ax[1].set_title('orginal')
        ax[2].imshow(gene)
        ax[2].set_title('generated')
        plt.show()


data,_=next(iter(dataloader_val))
samplenum=1
index=randint(0,len(data),samplenum)

satellite=data[index,:,:,:256].to(device)
maps=data[index,:,:,256:].to(device)


model=torch.load('torch_generator.pth')
model.to(device)

generatedImg=model(satellite)

data_visualtion_3part(generatedImg.cpu(),satellite.cpu(),maps.cpu(),samplenum)

