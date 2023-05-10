import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.UACANet import UACANet
from utils.dataloader import get_loader, test_dataset
# from lib.pvt import PolypPVT
# from lib.lraspp import lraspp_mobilenetv3_large
# from utils.dataloader import test_dataset
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
from sklearn.utils.multiclass import type_of_target
import time
#Y_train0为真实标签，Y_pred_0为预测标签，注意，这里roc_curve为一维的输入，Y_train0是一维的
def funcroc(gt,Y_pred_0):
    target = np.array(gt)
    input_flat = np.reshape(Y_pred_0, (-1)) 
    target_flat = np.reshape(target, (-1))
    # fpr, tpr, thresholds_keras = roc_curve(input_flat,target_flat,pos_label=0.5) 
    fpr, tpr, _ = roc_curve(input_flat.astype('int'),  target_flat)
    # print('tpr',len(tpr))
    # auc1 = auc(fpr, tpr)
    # # plt.figure()
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr, tpr, label='S3< val (AUC = {:.3f})'.format(auc))
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve')
    # plt.legend(loc='best')
    # plt.savefig('roc.png')
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='/home/hzy/xiefang/deeplearning/input/UACANet-lunwen/15PolypPVT-best.pth')
    opt = parser.parse_args()
    # model = PolypPVT()
    model=UACANet().cuda()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval() 
    torch.cuda.synchronize()
    start = time.time()
    for _data_name in [ 'Kvasir']:
        ##### put data_path here #####
        data_path = './data/TestDataset/{}'.format(_data_name)
        ##### save_path #####
        # save_path = './result_map/Bashline/{}/'.format(_data_name)
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, 352)
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            # gt = np.asarray(gt, np.float32)
            # gt /= (gt.max() + 1e-8)
            image = image.cuda()
            P1,p2,p3,p4 = model(image)
            # res = F.upsample(P1+p2+p3+p4, size=gt.shape, mode='bilinear', align_corners=False)
            # res = res.sigmoid().data.cpu().numpy().squeeze()
            # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # cv2.imwrite(save_path+name, res*255)
    torch.cuda.synchronize()
    end = time.time()
    print('infer_time:', end-start)
    print(_data_name, 'Finish!')
            # target = np.array(gt)
            # input_flat = np.reshape(res, (-1)) 
            # target_flat = np.reshape(target, (-1))
           
          # fpr, tpr, thresholds_keras = roc_curve(input_flat,target_flat,pos_label=0.5) 
            # fpr, tpr, _ = roc_curve(input_flat.astype('int'),  target_flat,pos_label=1)
        
            # ans1.append(list(fpr)[1])
            # ans2.append(list(tpr)[1])
            # auc1 = anc1.append(auc(fpr, tpr))
            # funcroc(gt,res)
            # cv2.imwrite(save_path+name, res*255)
        # # plt.figure()
        
        # plt.plot([0, 1], [0, 1], 'k--')
        # ans1.sort()
        # ans2.sort()
        # plt.plot(ans1, ans2)
        # plt.xlabel('False positive rate')
        # plt.ylabel('True positive rate')
        # plt.xlim(0,0.008)
        # plt.ylim(0.75,1)
        # plt.title('ROC curve')
        # plt.legend(loc='best')
        # plt.savefig('roc.png')
        # print(_data_name, 'Finish!')
