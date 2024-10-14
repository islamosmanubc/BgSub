import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.ops.focal_loss import sigmoid_focal_loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class FgSegLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.focalloss = FocalLoss(0)
    def iou_loss(self,pred, mask):
        
        inter = (pred*mask).sum(dim=(2,3))
        union = (pred+mask).sum(dim=(2,3))
        iou  = 1-(inter+1)/(union-inter+1)
        return iou.mean()
    
    def forward(self, pred, mask):
        pred = torch.sigmoid(pred)
        #loss = sigmoid_focal_loss(pred,mask)+self.iou_loss(pred,mask)
        l1 = F.mse_loss(pred, mask)
        l2 = self.iou_loss(pred, mask)
        loss  =  3*l1 + l2
        #lossb = F.binary_cross_entropy_with_logits(outb1, body)
        #lossd = F.binary_cross_entropy_with_logits(outd1, detail)
        #loss   = lossb1 + lossd1 + loss
        return loss
    
#def region_edge(datapath):
    #mask = cv2.imread(datapath+'/mask/'+name,0)
    #body = cv2.blur(mask, ksize=(5,5))
    #body = cv2.distanceTransform(body, distanceType=cv2.DIST_L2, maskSize=5)
    #body = body**0.5

    #tmp  = body[np.where(body>0)]
    #if len(tmp)!=0:
    #    body[np.where(body>0)] = np.floor(tmp/np.max(tmp)*255)

    #cv2.imwrite(datapath+'/body-origin/'+name, body)
    #cv2.imwrite(datapath+'/detail-origin/'+name, mask-body)