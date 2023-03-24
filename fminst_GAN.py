### -----------------import libraries-------------------###
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader,Subset

import sys

import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transformations
transform = T.Compose([ T.ToTensor(),
                        T.Resize(64),
                        T.Normalize(.5,.5),
                       ])

dataset = torchvision.datasets.FashionMNIST(root='./data', download=False, transform=transform)


keep_classes=['Pullover','Trouser','Dress','Coat','Sandal']

images_use=torch.Tensor()
for i in range(len(keep_classes)):
    classidx=dataset.classes.index(keep_classes[i])
    images_use=torch.cat((images_use,torch.where(dataset.targets==classidx)[0]),0 ).type(torch.long)
    
batchsize=100
sampler=torch.utils.data.sampler.SubsetRandomSampler(images_use)
data_loader=DataLoader(dataset,sampler=sampler,batch_size=batchsize,drop_last=True)

# data checking
dat,lab=next(iter(data_loader))
fig,axs=plt.subplots(2,5,figsize=(10,6))

for (i,ax) in enumerate(axs.flatten()):
    pic=torch.squeeze(dat.data[i])
    pic=pic/2+.5
    label=dataset.classes[lab[i]]
    
    ax.imshow(pic,cmap='gray')
    ax.text(14,0,label,ha='center',fontweight='bold',color='k',backgroundcolor='y')
    ax.axis('off')

plt.tight_layout()
plt.show()

###   --------------MODEL-------------------    ###
# GAN MODEL
class discriminator(nn.Module):
    def __init__(self,printShape=False):
        super().__init__()
        
        self.conv1=nn.Conv2d(1,64,4,2,1, bias=False)
        self.conv2=nn.Conv2d(64,128,4,2,1, bias=False)
        self.conv3=nn.Conv2d(128,256,4,2,1, bias=False)
        self.conv4=nn.Conv2d(256,512,4,2,1, bias=False)
        self.conv5=nn.Conv2d(512,1,4,1,0, bias=False)
        
        self.batch2=nn.BatchNorm2d(128)
        self.batch3=nn.BatchNorm2d(256)
        self.batch4=nn.BatchNorm2d(512)
        
        self.printShape=printShape
        
    def forward(self,x):
        if self.printShape is True: print(x.shape)
        
        x=F.leaky_relu(self.conv1(x),negative_slope=.2)
        if self.printShape is True: print(x.shape)
        
        x=F.leaky_relu(self.conv2(x),negative_slope=.2)
        x=self.batch2(x)
        if self.printShape is True: print(x.shape)
        
        x=F.leaky_relu(self.conv3(x), negative_slope=.2)
        x=self.batch3(x)
        if self.printShape is True: print(x.shape)
        
        x=F.leaky_relu(self.conv4(x), negative_slope=.2)
        x=self.batch4(x)
        if self.printShape is True: print(x.shape)
        
        return torch.sigmoid(self.conv5(x)).view(-1,1)


class generator(nn.Module):
    def __init__(self,printShape=False):
        super().__init__()
        
        self.conv1=nn.ConvTranspose2d( 100, 512, 4, 1, 0, bias=False)
        self.conv2=nn.ConvTranspose2d( 512, 256, 4, 2, 1, bias=False)
        self.conv3=nn.ConvTranspose2d( 256, 128, 4, 2, 1, bias=False)
        self.conv4=nn.ConvTranspose2d( 128, 64, 4, 2, 1, bias=False)
        self.conv5=nn.ConvTranspose2d(64, 16, 3, 1, 1, bias=False)
        self.conv6=nn.ConvTranspose2d( 16, 1, 4, 2, 1, bias=False)
        
        self.batch1=nn.BatchNorm2d(512)
        self.batch2=nn.BatchNorm2d(256)
        self.batch3=nn.BatchNorm2d(128)
        self.batch4=nn.BatchNorm2d(64)
        self.batch5=nn.BatchNorm2d(16)
        
        self.printShape=printShape
        
    def forward(self,x):
        if self.printShape is True: print(x.shape)
        
        x=F.relu(self.batch1( self.conv1(x)) )
        if self.printShape is True: print(x.shape)
        
        x=F.relu(self.batch2( self.conv2(x)) )
        if self.printShape is True: print(x.shape)
        
        x=F.relu(self.batch3( self.conv3(x)) )
        if self.printShape is True: print(x.shape)
        
        x=F.relu(self.batch4( self.conv4(x)) )
        if self.printShape is True: print(x.shape)
        
        x=F.relu(self.batch5( self.conv5(x)) )
        if self.printShape is True: print(x.shape)
        
        x=torch.tanh( self.conv6(x) )
        return x
 
# MODEL checking
DisNet=discriminator()
outDis=DisNet(torch.randn(5,1,64,64))
print(outDis)

GeneNet=generator(True)
out=GeneNet(torch.randn(5,100,1,1))  
print(out.shape)
plt.imshow(out[3,:,:,:].squeeze().detach().numpy())     


# train the Models
lossfun=nn.BCELoss()

disNet=discriminator().to(device)
genNet=generator().to(device)

disOptim=torch.optim.Adam(disNet.parameters(),lr=0.0002,betas=(.5,.999))
genOptim=torch.optim.Adam(genNet.parameters(),lr=.0002,betas=(.5,.999))

numEpoch=10

dislosses=torch.zeros(numEpoch)
genlosses=torch.zeros(numEpoch)

for epoch in range(numEpoch):
    for data,lab in data_loader:
        data=data.to(device)
        
        real_labels=torch.ones(batchsize,1).to(device)
        fake_labels=torch.zeros(batchsize,1).to(device)
        
        # ------discriminator-------
        pred_real=disNet(data) # prediction on real image
        dis_loss_real=lossfun(pred_real,real_labels)
        
        fake_data=torch.randn(batchsize,100,1,1).to(device)
        
        fake_image=genNet(fake_data)
        pred_fake=disNet(fake_image)
        dis_loss_fake=lossfun(pred_fake,fake_labels)
        
        d_loss=dis_loss_fake+dis_loss_real
        
        disOptim.zero_grad()
        d_loss.backward()
        disOptim.step()
      
        
        # ------- Generator --------
        fake_image=genNet(torch.randn(batchsize,100,1,1).to(device))
        pred_fake=disNet(fake_image)
        
        gen_loss=lossfun(pred_fake,real_labels)
        
        genOptim.zero_grad()
        gen_loss.backward()
        genOptim.step()
        
        dislosses[epoch]=d_loss
        genlosses[epoch]=gen_loss
    msg=f'epoch:{epoch+1} finsihed'
    sys.stdout.write('\r' + msg)
 

# showing loss values graph       
plt.plot(dislosses.detach())
plt.plot(genlosses.detach())
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend('discriminator','generator')


# generator model testing
genNet.eval()
fake_data = genNet( torch.randn(batchsize,100,1,1).to(device) ).cpu()

# and visualize...
fig,axs = plt.subplots(3,6,figsize=(12,6))
for i,ax in enumerate(axs.flatten()):
  ax.imshow(fake_data[i,:,].detach().squeeze(),cmap='gray')
  ax.axis('off')

plt.suptitle(keep_classes,y=.95,fontweight='bold')
plt.show()
