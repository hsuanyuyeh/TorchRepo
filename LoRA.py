import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from tqdm import tqdm
from torch.utils.data import DataLoader

_ = torch.manual_seed(42)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3801,))])
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(mnist_trainset, batch_size=10, shuffle=True)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

class SimpleNN(nn.Module):
    def __init__(self, hidden_size_1=1000, hidden_size_2=2000):
        super().__init__()
        self.linear1 = nn.Linear(28*28, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, 10)
        self.relu = nn.ReLU() 
    
    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

net = SimpleNN().to(device)

def train(model: nn.Module, 
          train_loader: DataLoader, 
          loss_fn: torch.nn.Module, 
          optimizer: torch.optim.Optimizer,
          device: torch.device):
    
    num_iterations = 0
    train_loss = 0
    model.to(device)
    for X, y in train_loader:
        num_iterations += 1
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        train_pred = model(X)
        loss = loss_fn(train_pred, y.type(torch.long))
        train_loss += loss
        loss.backward()
        optimizer.step()
    
    train_loss /= len(train_loader)
    print(f"Train Loss: {train_loss:.4f}")

def test(model: nn.Module, test_loader: DataLoader, device: torch.device):
    correct = 0
    total = 0
    model.to(device)
    with torch.inference_mode():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            for idx, i in enumerate(test_pred):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1
    
    print(f"Accuracy: {round(correct/total, 3)}")

# train a baseline model
# epochs=5
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# for epoch in tqdm(range(epochs)):
#     print(f"Epoch: {epoch} \n-----")
#     train(net, train_loader, loss_fn, optimizer, device)

original_weights = {}
for name, param in net.named_parameters():
    original_weights[name] = param.clone().detach() 

# introduce LoRA to improve the model
class LoRAParameterization(nn.Module):
    def __init__(self, features_in, features_out, rank, alpha, device):
        super().__init__()
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
        self.lora_A = nn.Parameter(torch.zeros((rank, features_out)).to(device))
        nn.init.normal(self.lora_A, mean=0, std=1)
        self.scale = alpha/rank
        self.enabled = True
    
    def forward(self, original_weights):
        if self.enabled:
            return original_weights + torch.matmul(self.lora_B, self.lora_A)*self.scale
        else:
            return original_weights
 

def linear_layer_parameterization(layer, device, rank=1, lora_alpha=1):
    features_in, features_out = layer.weight.shape
    return LoRAParameterization(features_in, features_out, rank, lora_alpha, device)


parametrize.register_parametrization(net.linear1, "weight", 
                                       linear_layer_parameterization(
                                           net.linear1, 
                                           device))
parametrize.register_parametrization(net.linear2, "weight",
                                       linear_layer_parameterization(
                                           net.linear2,
                                           device))
parametrize.register_parametrization(net.linear3, "weight",
                                       linear_layer_parameterization(
                                           net.linear3,
                                           device))

def enable_disable_lora(enabled=True):
    for layer in [net.linear1, net.linear2, net.linear3]:
        layer.parametrizations["weight"][0].enabled = enabled


# check how many parameters created by LoRA
total_parameters_lora = 0
total_parameters_non_lora = 0

for index, layer in enumerate([net.linear1, net.linear2, net.linear3]):
    total_parameters_lora += layer.parametrizations["weight"][0].lora_A.nelement()+layer.parametrizations["weight"][0].lora_B.nelement()
    total_parameters_non_lora += layer.weight.nelement() + layer.bias.nelement()
    print(f"Layer {index+1}: W: {layer.weight.shape} + B: {layer.bias.shape} + Lora_A: {layer.parametrizations['weight'][0].lora_A.shape} + Lora_B: {layer.parametrizations['weight'][0].lora_B.shape}")

print(f"Total number of parameters (original): {total_parameters_lora:,}")
print(f"Total number of parameters (original + LoRA): {total_parameters_lora + total_parameters_non_lora:,}")
print(f"Parameters introduced by LoRA: {total_parameters_lora:,}")
parameters_increments = (total_parameters_lora/total_parameters_non_lora) *100
print(f"Parameters increment: {parameters_increments:.3f}%")

# freeze the non-LoRA parameters
for name, param in net.named_parameters():
    if 'lora' not in name:
        print(f"Freezing non-LoRA parameter {name}")
        param.requires_grad = False

for name, param in net.named_parameters():
    print(name, param.requires_grad)

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# train the network with LoRA only 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = net.parameters(), lr=0.001)
epochs=1
train_loader = DataLoader(mnist_trainset, batch_size=10, shuffle=True)
for epoch in tqdm(range(epochs)):
    print("Epoch: {epoch} \n-----")
    train(net, train_loader, loss_fn, optimizer, device)

# check the frozen parameters weights and bias didn't change 
assert torch.all(net.linear1.parametrizations.weight.original == original_weights['linear1.weight'])
assert torch.all(net.linear2.parametrizations.weight.original == original_weights['linear2.weight'])
assert torch.all(net.linear3.parametrizations.weight.original == original_weights['linear3.weight'])
enable_disable_lora(enabled=True)
assert torch.equal(net.linear1.weight, net.linear1.parametrizations.weight.original + (net.linear1.parametrizations.weight[0].lora_B @ net.linear1.parametrizations.weight[0].lora_A) * net.linear1.parametrizations.weight[0].scale)
enable_disable_lora(enabled=False)
assert torch.equal(net.linear1.weight, original_weights['linear1.weight'])


# check the difference of non-LoRA vs LoRA inlcuded 
enable_disable_lora(enabled=True)
test(net, test_loader, device)

enable_disable_lora(enabled=False)
test(net, test_loader, device)
