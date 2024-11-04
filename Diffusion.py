import torch
import torchvision
from torch import nn
import torch.nn.functional as TF
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# data prep
IMG_SIZE=64
BATCH_SIZE=128
def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x:(x*2)-1)
    ]
    data_transform = transforms.Compose(data_transforms)
    train_data = torchvision.datasets.StanfordCars(root=".", 
                                                   download=False,
                                                   transform=data_transform)
    test_data = torchvision.datasets.StanfordCars(root=".",
                                                  download=False,
                                                  transform=data_transform,
                                                  split="test")
    #print(len(train_data), len(test_data))
    return torch.utils.data.ConcatDataset([train_data, test_data])
data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

"""
forward process - N(sqrt(1-betat)Xt-1, betatI)=N(sqrt(alphas)X0, (1-alphas)I)
alphas=alpha0*alpha1*...*alphat
Q(Xt|Xt-1) = sqrt(alphat)Xt-1 + sqrt(1-alphat)noise
"""
def linear_beta_schedule(timesteps, start=0.001, end=0.02):
    return torch.linspace(start, end, timesteps)

T = 300
betas = linear_beta_schedule(T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0-alphas_cumprod)



def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    vals=vals.to(device)
    t=t.to(device)
    outs = vals.gather(-1, t)
    return outs.reshape(batch_size, *((1,)*(len(x_shape)-1))).to(t.device)

def forward_diffusion_sample(x_0, t, device:torch.device):
    noice = torch.rand_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    return sqrt_alphas_cumprod_t.to(device)*x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device)*noice.to(device), noice.to(device)

# plot to inspect forward diffusion images
def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t+1)/2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t*255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage()
    ])
    if (len(image.shape) == 4):
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))

# simulate forward diffussion
image = next(iter(dataloader))[0]
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
num_images = 10
stepsize = int(T/num_images)
for idx in range(0, T, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    #plt.subplot(1, num_images+1, int(idx/stepsize)+1)
    img, nocie = forward_diffusion_sample(image, t, device)
    #show_tensor_image(img.cpu())
#plt.show()

"""
The backward process - Unet
Encoding path (down part): lowering the spatial resolution to capture high-resolution, low-level characteristics
Decoding path (up part): increasing the spatial resolution to constitute a dense segmentation map
"""
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, 
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        #Up part of UNET
        for feature in features[::-1]:
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(in_channels=features[0], 
                                    out_channels=out_channels,
                                    kernel_size=1)
    
    def forward(self, x):
        skip_connections=[]
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections=skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)
    
model = UNET()
print(model)

#train the model
def get_loss(model: nn.Module, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy)
    return TF.l1_loss(noise, noise_pred)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
epochs = 5
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
        loss = get_loss(model, batch[0], t)
        loss.backward()
        optimizer.step()

        
        print(f"Epoch: {epoch} | step: {step:03d}, Loss: {loss.item()}")
        






