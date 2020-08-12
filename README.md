# Spatial Feedback Learning to Improve Semantic Segmentation in Hot Weather
High-temperature weather conditions induce geometrical distortions in images which can adversely affect the performance of a computer vision model performing downstream tasks such as semantic segmentation. The performance of such models has been shown to improve by adding a restoration network before a semantic segmentation network. The restoration network removes the geometrical distortions from the images and shows improved segmentation results. However, this approach suffers from a major architectural drawback that is the restoration network does not learn directly from the errors of the segmentation network. In other words, the restoration network is not task aware. In this work, we propose a semantic feedback learning approach, which improves the task of semantic segmentation giving a feedback response into the restoration network. This response works as an attend and fix mechanism by focusing on those areas of an image where restoration needs improvement. Also, we proposed loss functions: Iterative Focal Loss (iFL) and Class-Balanced Iterative Focal Loss (CB-iFL), which are specifically designed to improve the performance of the feedback network. These losses focus more on those samples that are continuously miss-classified over successive iterations. Our approach gives a gain of 17.41 mIoU over the standard segmentation model, including the additional gain of 1.9 mIoU with CB-iFL on the Cityscapes dataset.
### Requirements
* PyTorch (version = 0.3.0)
* Python (version = 2.7)
* Jupyter Notebook
* numpy
* tqdm
* torchvision ==0.2.0

### Results

![Drag Racing](result.png)


### Contact
shyam.nandan@research.iiit.ac.in 
