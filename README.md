# ViTransPAD video transoformer for face anti-spoofing 

### [Paper ICIP2022](https://arxiv.org/pdf/2203.01562.pdf) | [Code](https://github.com/hengxyz/ViTransPAD/edit/main/README.md) | [Poster](https://drive.google.com/file/d/1P-xVT7uSp-SIu2yvRvfYiq3T0Pkqc9j4/view?usp=sharing) | [Slides](https://drive.google.com/file/d/1kS81q4-msA5JGNFv983QS0nEuBTeDJkm/view?usp=sharing) 

<br>VITRANSPAD: VIDEO TRANSFORMER USING CONVOLUTION AND SELF-ATTENTION FOR FACE PRESENTATION ATTACK DETECTION <br>


<!-- ---------------------------------------------- -->
## Citation
If any part of our paper and repository is helpful to your work, please generously cite with:
```
@article{ming2022vitranspad,
  title={ViTransPAD: Video Transformer using convolution and self-attention for Face Presentation Attack Detection},
  author={Ming, Zuheng and Yu, Zitong and Al-Ghadi, Musab and Visani, Muriel and MuzzamilLuqman, Muhammad and Burie, Jean-Christophe},
  journal={arXiv preprint arXiv:2203.01562},
  year={2022}
}
}
```

<!-- ---------------------------------------------- -->
## Introduction 
Face Presentation Attack Detection (PAD) is an important measure to prevent spoof attacks for face biometric systems.
Many works based on Convolution Neural Networks (CNNs) for face PAD formulate the problem as an image-level binary
classification task without considering the context. Alternatively, Vision Transformers (ViT) using self-attention to
attend the context of an image become the mainstreams in face PAD. Inspired by ViT, we propose a Video-based Transformer for face PAD (ViTransPAD) with short/long-range spatio-temporal attention which can not only focus on local details with short attention within a frame but also capture
long-range dependencies over frames. Instead of using coarse image patches with single-scale as in ViT, we propose the
Multi-scale Multi-Head Self-Attention (MsMHSA) architecture to accommodate multi-scale patch partitions of Q, K, V
feature maps to the heads of transformer in a coarse-to-fine manner, which enables to learn a fine-grained representation
to perform pixel-level discrimination for face PAD. Due to lack inductive biases of convolutions in pure transformers,
we also introduce convolutions to the proposed ViTransPAD to integrate the desirable properties of CNNs by using convolution patch embedding and convolution projection. The extensive experiments show the effectiveness of our proposed ViTransPAD with a preferable accuracy-computation
balance, which can serve as a new backbone for face PAD 
![Long-short attention](https://github.com/hengxyz/ViTransPAD/blob/main/figs/fig3_videoattention_cropped.jpg)

## Architecture: Multiscale Self-attetion CNN-Transformer 
![Long-short attention](https://github.com/hengxyz/ViTransPAD/blob/main/figs/fig1_architecture_multiSA.jpg)


## Visualization 
The attention map of Liveness and Attack frames in the video. The video-based attention map is more coherent to the image-wise transformer (such as Vit).  
![Long-short attention](https://github.com/hengxyz/ViTransPAD/blob/main/figs/visualisation.jpg)
 


<!-- ---------------------------------------------- -->
## Configuration & Training
Configutions for the parameters of the model: 
```
./configs/                                                
```


Training the model by running: 
```
./train.py -c configs/OuluNPU.json                                                   
```

<!-- ---------------------------------------------- -->
