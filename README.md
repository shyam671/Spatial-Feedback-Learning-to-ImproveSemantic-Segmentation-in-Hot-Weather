## Spatial Feedback Learning to Improve Semantic Segmentation in Hot Weather
High-temperature weather conditions induce geometrical distortions in images which can adversely affect the performance of a computer vision model performing downstream tasks such as semantic segmentation. The performance of such models has been shown to improve by adding a restoration network before a semantic segmentation network. The restoration network removes the geometrical distortions from the images and shows improved segmentation results. However, this approach suffers from a major architectural drawback that is the restoration network does not learn directly from the errors of the segmentation network. In other words, the restoration network is not task aware. In this work, we propose a semantic feedback learning approach, which improves the task of semantic segmentation giving a feedback response into the restoration network. This response works as an attend and fix mechanism by focusing on those areas of an image where restoration needs improvement. Also, we proposed loss functions: Iterative Focal Loss (iFL) and Class-Balanced Iterative Focal Loss (CB-iFL), which are specifically designed to improve the performance of the feedback network. These losses focus more on those samples that are continuously miss-classified over successive iterations. Our approach gives a gain of 17.41 mIoU over the standard segmentation model, including the additional gain of 1.9 mIoU with CB-iFL on the Cityscapes dataset.
## Requirements
* PyTorch (version = 0.3.0)
* Python (version = 2.7)
* Jupyter Notebook
* numpy
* tqdm
* torchvision ==0.2.0
* Matlab
## Code Structure and Summary

```  
| 
└───train
│   └───dataset.py         -- Dataloader script for Cityscapes
│   └───erfnet.py          -- Network structure for ErfNet
│   └───erfnet_imagenet.py -- Learning the initial weights for ErfNet
│   └───fill_weights.py    -- Class weights and ErfNet Network weights
│   └───iFL.py             -- Iterative focal loss for the feedback framework.
│   └───Interp.py          -- Layer that warps the input image to get final restored image.
│   └───iouEval.py         -- Evaluation of the model
│   └───main.py            -- Main driver programme
│   └───run_pretrained.py  -- Script to run the pre-trained models that outputs mIoU and segmentation results.
│   └───transform.py       -- Transform the segmentation output in original cityscapes color.
│   └───unet_model.py      -- Modified unet module that takes the feedback 
│   └───utils.py           -- Tranformation input and saving checkpoints
└───trained_models
│   └───erfnet_encoder_pretrained.pth.tar  --Pretrained Encoder
│   └───erfnet_pretrained.pth              --Pretrained Network
```
## Dataset Generation
* Download the cityscapes dataset [[Link]](https://www.cityscapes-dataset.com/) and simulation code in matlab from the [[Link]](https://webee.technion.ac.il/~yoav/research/turbulence_distortion.html)

* Run the matlab code using the parameter given in paper to generate the turbulent dataset.
## Training
```
usage: main.py      --datadir {path to dataset}
                    --height  {height of image (default = 224)}
                    --num-epochs {default = 150}
                    --batch-size {default = 8}                    
                    --savedir    {path to save}
                    --pretrainedEncoder {path to save}
optional arguments:
  --steps-loss 			{records loss after some number of steps (default=50)}
  --iouTrain        {stores the iou during training}
```
* Sample command
```
CUDA_VISIBLE_DEVICES=0,1 python main.py --savedir . --pretrainedEncoder '/path_to_model/trained_models/erfnet_encoder_pretrained.pth.tar'
```

## Pretrained Models
* Pre-Trained Restoration Model [[Model]](https://drive.google.com/file/d/1AJznWOOuKW8lR-q3_qbdn0OGOTXU8LBk/view?usp=sharing)
* Pre-Trained Segmentation Model [[Model]](https://drive.google.com/file/d/1_shJu5F9bR3FW5Df9Wt0D6915EVRzJfQ/view?usp=sharing)
```
python run_pretrained.py --savedir . --pretrainedEncoder '/path_to_model/trained_models/erfnet_encoder_pretrained.pth.tar'
```
## Results

![Drag Racing](results.png)


### Contact
shyam.nandan@research.iiit.ac.in 
