import torch

vitb8 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
print(vitb8)
print('khar')
checkpoint = torch.load('/home/alij/SSL-GraphNNCLR/dino_cifar10/dino_vitbase8_pretrain_full_checkpoint.pth', map_location='cpu')
print(checkpoint.keys())  # To verify it has 'model' key or similar
# For inspecting model state dict structure
for key, value in checkpoint['model'].items():
    print(key, value.shape)