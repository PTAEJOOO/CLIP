import os
import clip
import torch
from torchvision.datasets import CIFAR100

from random import random
from random import randrange


# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
# print(cifar100)

# Prepare the inputs
idx = randrange(10000)
print(idx)
image, class_id = cifar100[idx]

##
image.show()
print("\nTrue label in cifar100:")
print(cifar100.classes[class_id])
##

image_input = preprocess(image).unsqueeze(0).to(device)

## add color classes
color_list = ['red','black','blue','skyblue','green','orange','yellow','white','purple','gray','pink','navy','color']
color_cifar100_classes = cifar100.classes + color_list

text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in color_cifar100_classes]).to(device)
# print(text_inputs.shape)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(10)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{color_cifar100_classes[index]:>16s}: {100 * value.item():.2f}%")