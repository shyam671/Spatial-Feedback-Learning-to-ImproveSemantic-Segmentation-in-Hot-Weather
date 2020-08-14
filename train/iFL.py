import torch 
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable

def CB_iFl(logits, target, weights_vector, gamma, alpha, NUM_CLASSES = 20): 
  # Calculate log probabilities
  logp = F.log_softmax(logits, dim=1)

  # Gather log probabilities with respect to target
  logp = logp.gather(1, target.view(target.size(0), 1, target.size(1), target.size(2)))
  
  # for focal loss 
  pt = Variable(logp.data.exp())
  
  # Multiply with weights
  weights = target.float()

  # Assign weights
  for classnum in range(NUM_CLASSES):
      weights[weights == classnum] = weights_vector[classnum]

  weights = Variable(weights).view(target.size(0), 1, target.size(1), target.size(2))

  weighted_logp = (torch.pow((1-pt), gamma) * logp * weights).view(target.size(0), -1)

  # Rescale so that loss is in approx. same interval
  weighted_loss = weighted_logp.sum(1) / weights.view(target.size(0), -1).sum(1)

  # Average over mini-batch
  weighted_loss = -alpha*weighted_loss.mean()
  return weighted_loss


