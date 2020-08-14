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
    
    savedir = '/home/shyam.nandan/NewExp/final_code/save/' + args.savedir
    automated_log_path = savedir + "/automated_log.txt"
    modeltxtpath = savedir + "/model.txt"    

    if (not os.path.exists(automated_log_path)):    
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))

    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),eps=1e-08, weight_decay=2e-4)
    roptimizer = Adam(rmodel.parameters(), 2e-4, (0.9, 0.999))                                       

    start_epoch = 1
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    rscheduler = lr_scheduler.StepLR(roptimizer, step_size=30, gamma=0.5)                        
    
    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        scheduler.step()    
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

        model.train()
        for step, (timages, images, labels) in enumerate(loader):
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
                timages = timages.cuda()
           
            inputs = Variable(timages)
	    itargets = Variable(images)
            targets = Variable(labels)  
	    
	    ss_inputs = rmodel(inputs, flag = 0, r_fb1 = 0, r_fb2 = 0)
            
            outs = model(ss_inputs, only_encode=enc)

            tminus_outs = outs.detach()
            tplus_outs = outs.detach()
            
            outputs = []
            for num_feedback in range(3):
            	optimizer.zero_grad()
            	roptimizer.zero_grad()
                
                ss_inputs = rmodel(inputs, flag= 1, r_fb1 = (tplus_outs - tminus_outs) , r_fb2 = ss_inputs.detach())

                loss = rcriterion(ss_inputs, itargets)

                loss.backward()
                roptimizer.step()

            	optimizer.zero_grad()
            	roptimizer.zero_grad()
                   
            	outs = model(ss_inputs.detach(),only_encode=enc)
                
                outputs.append(outs)

                tminus_outs = tplus_outs
                tplus_outs = outs.detach()

            del outs, tminus_outs, tplus_outs
            gc.collect()
            
            loss = 0.0
            Gamma = [0, 0.1, 0.2]
            Alpha = [1, 1, 1]
            
            for i, o in enumerate(outputs):
                loss += CB_iFl(o, targets[:, 0], weight, gamma = Gamma[i], alpha = Alpha[i])
       
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data[0])
            time_train.append(time.time() - start_time)

            if (doIouTrain):
         
                iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
               
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
		print('loss:  ', average.data.cpu()[0], 'Epoch:  ', epoch, 'Step:  ', step)

        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)        
        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
            print ("EPOCH IoU on TRAIN set: ", iouStr, "%")  

        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        time_val = []

        if (doIouVal):
            iouEvalVal = iouEval(NUM_CLASSES)

        for step, (timages, images, labels) in enumerate(loader_val):
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
		timages = timages.cuda()

            inputs = Variable(timages, volatile=True)   
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

            	outs = model(ss_inputs.detach(),only_encode=enc)

                tminus_outs = tplus_outs

                tplus_outs = outs.detach()
  
            ##################################

            del ss_inputs, tplus_outs, tminus_outs
            outputs = outs
            loss = CB_iFl(outputs, targets[:, 0], weight, gamma = Gamma[0], alpha = Alpha[0])
            epoch_loss_val.append(loss.data[0])
            time_val.append(time.time() - start_time)

            if (doIouVal):
                iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
		print('Val loss:  ', average, 'Epoch:  ', epoch, 'Step:  ', step)

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)

        iouVal = 0
        if (doIouVal):
            iouVal, iou_classes = iouEvalVal.getIoU()
            iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
            print ("EPOCH IoU on VAL set: ", iouStr, "%") 
           
        # remember best valIoU and save checkpoint
        if iouVal == 0:
            current_acc = -average_epoch_loss_val
        else:
            current_acc = iouVal 

        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)

        filenameCheckpoint = savedir + '/checkpoint.pth.tar'
        filenameBest = savedir + '/model_best.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        #SAVE MODEL AFTER EPOCH
        filename = savedir + '/model-{epoch:03}.pth'
        filenamebest = savedir + '/model_best.pth'

        if args.epochs_save > 0 and step > 0 and step % args.epochs_save == 0:
            torch.save(model.state_dict(), filename)
            print(filename, epoch)
        if (is_best):
            torch.save(model.state_dict(), filenamebest)
            torch.save(rmodel.state_dict(), savedir + '/rmodel_best.pth')
            print(filenamebest,epoch)
            with open(savedir + "/best.txt", "w") as myfile:
                 myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))            

        #SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        #Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr ))
    
    return(model)


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
    model = train(args, rmodel, model, False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--model', default="/home/shyam.nandan/NewExp/final_code/train/erfnet") #change path here
    parser.add_argument('--datadir', default="/home/shyam.nandan/NewExp/cityscapes/") #change path here
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument('--epochs-save', type=int, default=0) 
    parser.add_argument('--savedir', required=True)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--pretrainedEncoder') 
    parser.add_argument('--iouTrain', action='store_true', default=False)
    parser.add_argument('--iouVal', action='store_true', default=True)  
    main(parser.parse_args())
