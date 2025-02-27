In the first notebook, I implemented a custom **DeepLabv3-like** architecture inspired by **ResNet**, but without loading pretrained weights. I modified the model to accommodate all **12 channels** of my dataset, which seemed to yield **improved results**.  

In the **V2 notebook**, I utilized the actual **ResNet** model with pretrained weights, but limited the input to only **3 channels** (RGB). The performance was **comparable** to the custom model, despite using fewer input channels.
