import numpy as np
from PIL import Image
import torch
from torch import nn
from hubconf import detr_resnet50, detr_resnet50_panoptic
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False);

# model path
path = './detector/DETR/checkpoint.pth'

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing, they are for preparation
'''
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)
'''

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = out_bbox
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).cuda()
    b = b.cpu()
    return b

# define detr class for detection
class DETR(object):
    def __init__(self):
        #detr = detr_resnet50(pretrained=False,num_classes=91).eval()
        #state_dict =  torch.load(path)   # <-----------修改加载模型的路径
        #detr.load_state_dict(state_dict["model"])
        #self.net = detr.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        #self.net = torch.load(path)
        self.net = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
    def __call__(self, ori_img, conf_threshold):
        assert isinstance(ori_img, np.ndarray), "input must be a numpy array!"
        # mean-std normalize the input image (batch-size: 1)
        img_pil = Image.fromarray(ori_img)
        img = transform(img_pil).unsqueeze(0)
        # propagate through the detr network
        img = img.cuda()
        outputs = self.net.cuda()(img)
        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]    
        keep = probas.max(-1).values > conf_threshold
        probas2 = probas[keep]
        bboxes2 = outputs['pred_boxes'][0,keep]
        confidence, class_id = torch.max(probas2, 1)
        # convert boxes from [0; 1] to image scales
        height, weight, layers = ori_img.shape
        bboxes_scaled = rescale_bboxes(bboxes2, (weight, height))
        # bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0], img[0][0].size())
        # assert False, print(bboxes2, ori_img.shape[:-1], img.shape)
        return bboxes_scaled, confidence, class_id
        
        
        
        
        
        