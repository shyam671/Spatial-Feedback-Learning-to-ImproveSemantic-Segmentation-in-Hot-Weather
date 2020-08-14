import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png")

def image_path(root, basename, extension):
    return os.path.join(root, basename, extension)

def image_path_city(root, name):
    return os.path.join(root, name)

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class cityscapes(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.timages_root = os.path.join('/home/shyam.nandan/NewExp/tcityscapes', 'leftImg8bit/') #change path here
        self.labels_root = os.path.join(root, 'gtFine/')
        
        self.images_root += subset
        self.timages_root += subset
        self.labels_root += subset

        print (self.images_root)
        
	self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

	self.tfilenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.timages_root)) for f in fn if is_image(f)]
        self.tfilenames.sort()
	
        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
	tfilename = self.tfilenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')

	with open(image_path_city(self.timages_root, tfilename), 'rb') as f:
            timage = load_image(f).convert('RGB')

        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        if self.co_transform is not None:
            image, timage, label = self.co_transform(image, timage, label)

	return timage, image, label, filename	
        #return timage, image, label

    def __len__(self):
        return len(self.filenames)

