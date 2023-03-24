 ###-----satellite image to maps translation using pix2pix GAN--------###

### --------------------libraries--------------------###
##-----torch------##
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

##---torchvision---##
import torchvision
from torchvision import models,transforms,datasets

from torchsummary import summary


import matplotlib.pyplot as plt
import numpy as np
import os


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
        
        
generator=create_generator()        
generator
































