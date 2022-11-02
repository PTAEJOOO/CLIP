import os
import clip
import torch
from torchvision.datasets import CIFAR100

import tensorflow
import tensorflow_datasets as tfds

from PIL import Image
import json
from random import random
from random import randrange

builder = tfds.builder('imagenet2012_real',data_dir="C:/Users/qkrxo/anaconda3/envs/clip/Lib/site-packages/tensorflow_datasets/")
builder.download_and_prepare()
data = builder.as_dataset(split='validation')

data_iter = data.as_numpy_iterator()
# print(data_iter)

num = randrange(50000)
print(num)
for i in range(num):
    example = data_iter.next()

##
# while True:
#     example = data_iter.next()
#     if len(example['real_label']) == 3:
#         break
##

## imagenet 1000 classes labels
with open("C:/Users/qkrxo/kist/CLIP/imagenet1000_clsidx_to_labels.txt", 'r') as labels:
    strings = labels.read()
    dicts = dict(eval(strings))
    imagenet_classes = dicts.values()
    imagenet_classes = list(imagenet_classes)

# ## simplifed imagenet 1000 classes labels
# with open('C:/Users/qkrxo/kist/CLIP/imagenet-simple-labels.json') as f:
#     imagenet_classes = json.load(f)

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Prepare the inputs
image = Image.fromarray(example['image'])
class_id = example['original_label']
real_id = example['real_label']

image.show()
print("\nSingle label in Imagenet:")
print(imagenet_classes[class_id])

print("\nReal labels in Imagenet:")
real_class=[]
for i in real_id:
    real_class.append(imagenet_classes[i])
print(real_class)

image_input = preprocess(image).unsqueeze(0).to(device)

extra_list = ['red','black','blue','skyblue','green','orange','yellow','white','purple','gray','pink','navy','color','man','woman']
imagenet_classes += extra_list
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in imagenet_classes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{imagenet_classes[index]:>16s}: {100 * value.item():.2f}%")