import torch
import timm

NUM_FINETUNE_CLASSES = 10
imgs = torch.randn(2, 3, 224, 224)
m0 = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=NUM_FINETUNE_CLASSES)
m1 = timm.create_model('vit_base_patch16_224', pretrained=True)
logits00 = m0(imgs) # 10 classes
logits01 = m1(imgs) # 1000 classes default classes as same as imagenet
print(f'logits00 shape: {logits00.shape}')
print(f'logits01 shape: {logits01.shape}')
embeddings10 = m0.forward_features(imgs)
embeddings11 = m1.forward_features(imgs)
print(f'embeddings10 shape: {embeddings10.shape}')
print(f'embeddings11 shape: {embeddings11.shape}')
