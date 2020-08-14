import torch 
from PIL import Image
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from transform import Relabel, ToLabel, Colorize

class MyCoTransform(object):
    def __init__(self, augment=True, height=224):
        self.augment = augment
        self.height = height
        pass
    def __call__(self, input, tinput, target):
        input =  Resize(self.height, Image.BILINEAR)(input)
        tinput =  Resize(self.height, Image.BILINEAR)(tinput)
        target = Resize(self.height, Image.NEAREST)(target)

        input = ToTensor()(input)
        tinput = ToTensor()(tinput)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, tinput, target

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)

