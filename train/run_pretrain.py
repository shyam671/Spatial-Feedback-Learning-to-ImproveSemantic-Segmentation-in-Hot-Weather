import os
import gc
import time
import torch
import math
import random
import numpy as np

from iFL import CB_iFl
from erfnet import Net
from unet_model import UNet
from shutil import copyfile
from dataset import cityscapes
from PIL import Image, ImageOps
from argparse import ArgumentParser
from torch.autograd import Variable
from torch.utils.data import DataLoader
from iouEval import iouEval, getColorEntry
from torch.optim import SGD, Adam, lr_scheduler
from transform import Relabel, ToLabel, Colorize
from utils import MyCoTransform, save_checkpoint
from fill_weights import fill_weights, classWeights
from erfnet_imagenet import ERFNet as ERFNet_imagenet
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad, ToTensor, ToPILImage

NUM_CHANNELS = 3
NUM_CLASSES = 20 

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()

def train(args, rmodel, model, enc=False):
    best_acc = 0
    weight = classWeights(NUM_CLASSES)
    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded"

    co_transform = MyCoTransform(augment=True, height=args.height)
    co_transform_val = MyCoTransform(augment=False, height=args.height)
    dataset_train = cityscapes(args.datadir, co_transform, 'train')
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')

    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    if args.cuda:
        weight = weight.cuda()
    rcriterion = torch.nn.L1Loss()
    

    savedir = '/home/shyam.nandan/NewExp/F_erfnet_pytorch_ours_w_gt_v2_multiply/save/' + args.savedir #change path

    if (enc):
        automated_log_path = savedir + "/automated_log_encoder.txt"
        modeltxtpath = savedir + "/model_encoder.txt"
    else:
        automated_log_path = savedir + "/automated_log.txt"
        modeltxtpath = savedir + "/model.txt"    

    if (not os.path.exists(automated_log_path)):     
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))

    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),eps=1e-08, weight_decay=2e-4)##  
    roptimizer = Adam(rmodel.parameters(), 2e-4, (0.9, 0.999))                                  ## restoration scheduler     

    start_epoch = 1
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    rscheduler = lr_scheduler.StepLR(roptimizer, step_size=30, gamma=0.5)                        ## Restoration schedular 
    


    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        scheduler.step()    ## scheduler 2
        rscheduler.step()
        
	epoch_loss = []
        time_train = []
     
        doIouTrain = args.iouTrain   
        doIouVal =  args.iouVal      

        if (doIouTrain):
            iouEvalTrain = iouEval(NUM_CLASSES)

        usedLr = 0
        rusedLr = 0
        for param_group in optimizer.param_groups:
            print("Segmentation LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])
        for param_group in roptimizer.param_groups:
            print("Restoration LEARNING RATE: ", param_group['lr'])
            rusedLr = float(param_group['lr'])

        
        model.eval()
        epoch_loss_val = []
        time_val = []

        if (doIouVal):
            iouEvalVal = iouEval(NUM_CLASSES)

        for step, (timages, images, labels, filename) in enumerate(loader_val):
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
		timages = timages.cuda()

            inputs = Variable(timages, volatile=True)    #volatile flag makes it free backward or outputs for eval
            itargets = Variable(images, volatile=True)
	    targets = Variable(labels, volatile=True)
	    
	    ss_inputs = rmodel(inputs, flag = 0, r_fb1 = 0, r_fb2 = 0)


            outs = model(ss_inputs, only_encode=enc)
            
            tminus_outs = outs.detach()
            tplus_outs = outs.detach()

            
            for num_feedback in range(3):

            	optimizer.zero_grad()
            	roptimizer.zero_grad()
                
                ss_inputs = rmodel(inputs, flag= 1, r_fb1 = (tplus_outs - tminus_outs) , r_fb2 = ss_inputs.detach())

                loss = rcriterion(ss_inputs, itargets)
 
            	outs = model(ss_inputs.detach(), only_encode=enc)

                tminus_outs = tplus_outs
                tplus_outs = outs.detach()
  
            outputs = outs
            del outs, tminus_outs, tplus_outs
            gc.collect()
            Gamma = [0,0,0]
            Alpha = [1, 1, 1]
            loss = CB_iFl(outputs, targets[:, 0], weight, gamma = Gamma[0], alpha = Alpha[0])
            epoch_loss_val.append(loss.data[0])
            time_val.append(time.time() - start_time)


            if (doIouVal):
                #start_time_iou = time.time()
		iouEvalVal_img = iouEval(NUM_CLASSES)                
		iouEvalVal_img.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

                iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

                #print ("Time to add confusion matrix: ", time.time() - start_time_iou)
                label_color = Colorize()(outputs[0].max(0)[1].byte().cpu().data.unsqueeze(0))
                label_save = ToPILImage()(label_color)   
                 
                filenameSave = '../save_color_restored_joint_afl_CBFL/' + filename[0].split('/')[-2]

                im_iou, _ = iouEvalVal_img.getIoU()

    		if not os.path.exists(filenameSave):
        	      os.makedirs(filenameSave)
                #Uncomment to save output
                #label_save.save(filenameSave+ '/' + str(" %6.4f " %im_iou[0].data.numpy()) + '_' + filename[0].split('/')[-1])

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
		print('Val loss:  ', average, 'Epoch:  ', epoch, 'Step:  ', step)

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
        

        iouVal = 0
        if (doIouVal):
            iouVal, iou_classes = iouEvalVal.getIoU()
            iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
            print (iouVal, iou_classes, iouStr) 
  
    return(model)   

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)


def main(args):
    
    savedir = '/home/shyam.nandan/NewExp/final_code/save/' + args.savedir #change path here 

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    rmodel = UNet()
    rmodel = torch.nn.DataParallel(rmodel).cuda()
    pretrainedEnc = torch.nn.DataParallel(ERFNet_imagenet(1000))
    pretrainedEnc.load_state_dict(torch.load(args.pretrainedEncoder)['state_dict'])
    pretrainedEnc = next(pretrainedEnc.children()).features.encoder
    model = Net(NUM_CLASSES) 
    model = fill_weights(model, pretrainedEnc)
    model = torch.nn.DataParallel(model).cuda()
    #model = train(args, rmodel, model, False)

    PATH = '/home/shyam.nandan/NewExp/final_code/results/CB_iFL/rmodel_best.pth'
    rmodel.load_state_dict(torch.load(PATH))    

    PATH = '/home/shyam.nandan/NewExp/final_code/results/CB_iFL/model_best.pth'

    model.load_state_dict(torch.load(PATH))

    model = train(args, rmodel, model, False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)  
    parser.add_argument('--model', default="/home/shyam.nandan/NewExp/erfnet_pytorch_ours_w_gt_v2_multiply/train/erfnet")
    parser.add_argument('--state')

    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--datadir', default="/home/shyam.nandan/NewExp/cityscapes/")
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=0)    
    parser.add_argument('--savedir', required=True)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--pretrainedEncoder') 
    parser.add_argument('--visualize', action='store_true')

    parser.add_argument('--iouTrain', action='store_true', default=False) 
    parser.add_argument('--iouVal', action='store_true', default=True)  
    parser.add_argument('--resume', action='store_true')    #Use this flag to load last checkpoint for training  

    main(parser.parse_args())
