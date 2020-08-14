import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def classWeights(NUM_CLASSES):
    cb_beta = torch.Tensor([0.9999])
    weight = torch.ones(NUM_CLASSES)
    weight[0] = (1 - cb_beta)/(1 - torch.pow(cb_beta, 100/2.8149201869965))	
    weight[1] = (1 - cb_beta)/(1 - torch.pow(cb_beta, 100/6.9850029945374))	
    weight[2] = (1 - cb_beta)/(1 - torch.pow(cb_beta, 100/3.7890393733978))	
    weight[3] = (1 - cb_beta)/(1 - torch.pow(cb_beta, 100/9.9428062438965))	
    weight[4] = (1 - cb_beta)/(1 - torch.pow(cb_beta, 100/9.7702074050903))	
    weight[5] = (1 - cb_beta)/(1 - torch.pow(cb_beta, 100/9.5110931396484))	
    weight[6] = (1 - cb_beta)/(1 - torch.pow(cb_beta, 100/10.311357498169))	
    weight[7] = (1 - cb_beta)/(1 - torch.pow(cb_beta, 100/10.026463508606))	
    weight[8] = (1 - cb_beta)/(1 - torch.pow(cb_beta, 100/4.6323022842407))	
    weight[9] = (1 - cb_beta)/(1 - torch.pow(cb_beta, 100/9.5608062744141))	
    weight[10] = (1 - cb_beta)/(1 - torch.pow(cb_beta, 100/7.8698215484619))	
    weight[11] = (1 - cb_beta)/(1 - torch.pow(cb_beta, 100/9.5168733596802))	
    weight[12] = (1 - cb_beta)/(1 - torch.pow(cb_beta, 100/10.373730659485))	
    weight[13] = (1 - cb_beta)/(1 - torch.pow(cb_beta, 100/6.6616044044495))	
    weight[14] = (1 - cb_beta)/(1 - torch.pow(cb_beta, 100/10.260489463806))	
    weight[15] = (1 - cb_beta)/(1 - torch.pow(cb_beta, 100/10.287888526917))	
    weight[16] = (1 - cb_beta)/(1 - torch.pow(cb_beta, 100/10.289801597595))	
    weight[17] = (1 - cb_beta)/(1 - torch.pow(cb_beta, 100/10.405355453491))	
    weight[18] = (1 - cb_beta)/(1 - torch.pow(cb_beta, 100/10.138095855713))	
    weight[19] = 0
    return weight


def fill_weights(model, pretrainedEnc):
    #initial_block
    model.encoder.initial_block.conv.weight = pretrainedEnc.initial_block.conv.weight
    model.encoder.initial_block.conv.bias = pretrainedEnc.initial_block.conv.bias
    model.encoder.initial_block.bn.weight = pretrainedEnc.initial_block.bn.weight
    model.encoder.initial_block.bn.bias = pretrainedEnc.initial_block.bn.bias
    model.encoder.initial_block.bn.running_mean = pretrainedEnc.initial_block.bn.running_mean
    model.encoder.initial_block.bn.running_var = pretrainedEnc.initial_block.bn.running_var
    
    model.encoder.layers[0].conv.weight  = pretrainedEnc.layers[0].conv.weight 
    model.encoder.layers[0].conv.bias    = pretrainedEnc.layers[0].conv.bias 
    model.encoder.layers[0].bn.weight  = pretrainedEnc.layers[0].bn.weight
    model.encoder.layers[0].bn.bias = pretrainedEnc.layers[0].bn.bias 
    model.encoder.layers[0].bn.running_mean = pretrainedEnc.layers[0].bn.running_mean 
    model.encoder.layers[0].bn.running_var = pretrainedEnc.layers[0].bn.running_var 

    model.encoder.layers[6].conv.weight  = pretrainedEnc.layers[6].conv.weight 
    model.encoder.layers[6].conv.bias    = pretrainedEnc.layers[6].conv.bias 
    model.encoder.layers[6].bn.weight  = pretrainedEnc.layers[6].bn.weight
    model.encoder.layers[6].bn.bias = pretrainedEnc.layers[6].bn.bias 
    model.encoder.layers[6].bn.running_mean = pretrainedEnc.layers[6].bn.running_mean 
    model.encoder.layers[6].bn.running_var = pretrainedEnc.layers[6].bn.running_var 

    #Non bottelneck block
    for i in range(1,6):
    	model.encoder.layers[i].conv3x1_1.weight = pretrainedEnc.layers[i].conv3x1_1.weight 
    	model.encoder.layers[i].conv3x1_1.bias = pretrainedEnc.layers[i].conv3x1_1.bias
    	model.encoder.layers[i].conv1x3_1.weight = pretrainedEnc.layers[i].conv1x3_1.weight 
    	model.encoder.layers[i].conv1x3_1.bias = pretrainedEnc.layers[i].conv1x3_1.bias 
    	model.encoder.layers[i].bn1.weight = pretrainedEnc.layers[i].bn1.weight 
    	model.encoder.layers[i].bn1.bias = pretrainedEnc.layers[i].bn1.bias 
    	model.encoder.layers[i].bn1.running_mean = pretrainedEnc.layers[i].bn1.running_mean 
    	model.encoder.layers[i].bn1.running_var = pretrainedEnc.layers[i].bn1.running_var
    	model.encoder.layers[i].conv3x1_2.weight = pretrainedEnc.layers[i].conv3x1_2.weight 
    	model.encoder.layers[i].conv3x1_2.bias = pretrainedEnc.layers[i].conv3x1_2.bias 
    	model.encoder.layers[i].conv1x3_2.weight = pretrainedEnc.layers[i].conv1x3_2.weight 
    	model.encoder.layers[i].conv1x3_2.bias = pretrainedEnc.layers[i].conv1x3_2.bias 
    	model.encoder.layers[i].bn2.weight = pretrainedEnc.layers[i].bn2.weight 
    	model.encoder.layers[i].bn2.bias = pretrainedEnc.layers[i].bn2.bias 
    	model.encoder.layers[i].bn2.running_mean = pretrainedEnc.layers[i].bn2.running_mean 
    	model.encoder.layers[i].bn2.running_var = pretrainedEnc.layers[i].bn2.running_var

    for i in range(0,8):
    	model.encoder.f_layers[i].conv3x1_1.weight = pretrainedEnc.layers[i+7].conv3x1_1.weight 
    	model.encoder.f_layers[i].conv3x1_1.bias = pretrainedEnc.layers[i+7].conv3x1_1.bias
    	model.encoder.f_layers[i].conv1x3_1.weight = pretrainedEnc.layers[i+7].conv1x3_1.weight 
    	model.encoder.f_layers[i].conv1x3_1.bias = pretrainedEnc.layers[i+7].conv1x3_1.bias 
    	model.encoder.f_layers[i].bn1.weight = pretrainedEnc.layers[i+7].bn1.weight 
    	model.encoder.f_layers[i].bn1.bias = pretrainedEnc.layers[i+7].bn1.bias 
    	model.encoder.f_layers[i].bn1.running_mean = pretrainedEnc.layers[i+7].bn1.running_mean 
    	model.encoder.f_layers[i].bn1.running_var = pretrainedEnc.layers[i+7].bn1.running_var
    	model.encoder.f_layers[i].conv3x1_2.weight = pretrainedEnc.layers[i+7].conv3x1_2.weight 
    	model.encoder.f_layers[i].conv3x1_2.bias = pretrainedEnc.layers[i+7].conv3x1_2.bias 
    	model.encoder.f_layers[i].conv1x3_2.weight = pretrainedEnc.layers[i+7].conv1x3_2.weight 
    	model.encoder.f_layers[i].conv1x3_2.bias = pretrainedEnc.layers[i+7].conv1x3_2.bias 
    	model.encoder.f_layers[i].bn2.weight = pretrainedEnc.layers[i+7].bn2.weight 
    	model.encoder.f_layers[i].bn2.bias = pretrainedEnc.layers[i+7].bn2.bias 
    	model.encoder.f_layers[i].bn2.running_mean = pretrainedEnc.layers[i+7].bn2.running_mean 
    	model.encoder.f_layers[i].bn2.running_var = pretrainedEnc.layers[i+7].bn2.running_var    
 
    return model















