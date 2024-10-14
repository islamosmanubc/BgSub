import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from torch.nn import Softmax
import os
inv_im_trans = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225])

def store_samples(x,yh,bg,y,it):
    x = inv_im_trans(x).permute(0,2,3,1)
    xe = inv_im_trans(bg).permute(0,2,3,1)
    cx = x[-1].detach().cpu().numpy()
    yd = cx*255
    d = np.uint8(yd)
    im = Image.fromarray(d)
    im.save('metric_depth/visualization/'+str(it)+'_x.png')
    cex = xe[-1].detach().cpu().numpy()
    yd = cex*255
    d = np.uint8(yd)
    im = Image.fromarray(d)
    im.save('metric_depth/visualization/'+str(it)+'_xe.png')

    cy = y[-1].detach().cpu().numpy()
    d = np.zeros((224,224,3))
    yd = cy
    d[:,:,0] = yd*255
    d[:,:,1] = yd*255
    d[:,:,2] = yd*255
    d = np.uint8(d)
    im = Image.fromarray(d)
    im.save('metric_depth/visualization/'+str(it)+'_y.png')

    yh = torch.sigmoid(yh)
    cyh = yh[-1].detach().cpu().numpy()
    #cyh[cyh>0.5] = 1
    #cyh[cyh<=0.5] = 0
    yd = cyh
    d = np.zeros((224,224,3))
    yd = np.reshape(yd, (224,224))
    yd = np.double(yd)
    d[:,:,0] = yd*255
    d[:,:,1] = yd*255
    d[:,:,2] = yd*255
    d = np.uint8(d)
    im = Image.fromarray(d)
    im.save('metric_depth/visualization/'+str(it)+'_yh.png')


def store_vid_samples(x,yh,bg,gt,it,dataset,vid):
    x = inv_im_trans(x).permute(0,2,3,1)
    xe = inv_im_trans(bg).permute(0,2,3,1)
    for k in range(len(x)):
        save_path = 'CDNet/'+dataset[k]+'/'+vid[k]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cx = x[k].detach().cpu().numpy()
        yd = cx*255
        d = np.uint8(yd)
        im = Image.fromarray(d)
        im.save(save_path+'/'+str(it)+'_'+str(k)+'_x.png')
        cex = xe[k].detach().cpu().numpy()
        yd = cex*255
        d = np.uint8(yd)
        im = Image.fromarray(d)
        im.save(save_path+'/'+str(it)+'_'+str(k)+'_xe.png')

        cy = gt[k].detach().cpu().numpy()
        d = np.zeros((224,224,3))
        yd = cy
        d[:,:,0] = yd*255
        d[:,:,1] = yd*255
        d[:,:,2] = yd*255
        d = np.uint8(d)
        im = Image.fromarray(d)
        im.save(save_path+'/'+str(it)+'_'+str(k)+'_y.png')

        yh = torch.sigmoid(yh)
        cyh = yh[k].detach().cpu().numpy()
        cyh[cyh>0.5] = 1
        cyh[cyh<=0.5] = 0
        yd = cyh
        d = np.zeros((224,224,3))
        yd = np.reshape(yd, (224,224))
        yd = np.double(yd)
        d[:,:,0] = yd*255
        d[:,:,1] = yd*255
        d[:,:,2] = yd*255
        d = np.uint8(d)
        im = Image.fromarray(d)
        im.save(save_path+'/'+str(it)+'_'+str(k)+'_yh.png')


def store_all_samples(x,yh,bg,it):
    x = inv_im_trans(x).permute(0,2,3,1)
    xe = inv_im_trans(bg).permute(0,2,3,1)
    for k in range(len(x)):
        cx = x[k].detach().cpu().numpy()
        yd = cx*255
        d = np.uint8(yd)
        im = Image.fromarray(d)
        im.save('metric_depth/sample_results/'+str(it)+'_'+str(k)+'_x.png')
        cex = xe[k].detach().cpu().numpy()
        ydd = cex*255
        ydg = yd[:,:,0]+yd[:,:,1]+yd[:,:,2]
        ydg /= 3
        yddg = ydd[:,:,0]+ydd[:,:,1]+ydd[:,:,2]
        yddg /= 3
        yd = np.abs(ydg - yddg)
        d = np.uint8(yd)
        d[d>20] = 255
        d[d<=20] = 0
        dd = np.zeros((224,224,3))
        dd[:,:,0] = d
        dd[:,:,1] = d
        dd[:,:,2] = d
        dd = np.uint8(dd)

        im = Image.fromarray(dd)
        im.save('metric_depth/sample_results/'+str(it)+'_'+str(k)+'_xe.png')

        yh = torch.sigmoid(yh)
        cyh = yh[k].detach().cpu().numpy()
        cyh[cyh>0.7] = 1
        cyh[cyh<=0.7] = 0
        yd = cyh
        d = np.zeros((224,224,3))
        yd = np.reshape(yd, (224,224))
        yd = np.double(yd)
        d[:,:,0] = yd*255
        d[:,:,1] = yd*255
        d[:,:,2] = yd*255
        d = np.uint8(d)
        im = Image.fromarray(d)
        im.save('metric_depth/sample_results/'+str(it)+'_'+str(k)+'_yh.png')


def store_one_samples(x,xe,ye,y,yhat,num_obj,it,frame,task,th=0.5):
    if task == 'VideoObjectSegmentation':
        x = inv_im_trans(x).permute(0,2,3,1)
        xe = inv_im_trans(xe).permute(0,2,3,1)
        cx = x[-1].detach().cpu().numpy()
        yd = cx*255
        d = np.uint8(yd)
        im = Image.fromarray(d)
        im.save('visualization/stored_examples3/'+str(it)+'_'+str(frame)+'_x.png')
        cex = xe[-1].detach().cpu().numpy()
        yd = cex*255
        d = np.uint8(yd)
        im = Image.fromarray(d)
        im.save('visualization/stored_examples3/'+str(it)+'_'+str(frame)+'_xe.png')
        if num_obj == 1:
            obj = 0
        else:
            obj = np.random.randint(0,num_obj-1)

        single_frame = ye[-1,0:num_obj]
        single_frame[single_frame<=th] = 0
        new_frame = torch.zeros((num_obj+1,224,224))+0.1
        new_frame[1:] = single_frame
        new_frame = 10 - torch.argmax(new_frame,dim=0)
        new_frame[new_frame==10] = 0
        ye = new_frame
        ye = ye.detach().cpu().numpy()
        yd = ye
        d = np.zeros((224,224,3))
        yd = np.reshape(yd, (224,224))
        yd = np.double(yd)
        d[:,:,0] = yd*25
        d[:,:,1] = yd*25
        d[:,:,2] = yd*25
        d = np.uint8(d)
        im = Image.fromarray(d)
        im.save('visualization/stored_examples3/'+str(it)+'_'+str(frame)+'_yeobjects.png')


        single_frame = y[-1,0:num_obj]
        single_frame[single_frame<=th] = 0
        new_frame = torch.zeros((num_obj+1,128,128))+0.1
        new_frame[1:] = single_frame
        new_frame = 10 - torch.argmax(new_frame,dim=0)
        new_frame[new_frame==10] = 0
        y = new_frame
        y = y.detach().cpu().numpy()
        yd = y
        d = np.zeros((128,128,3))
        yd = np.reshape(yd, (128,128))
        yd = np.double(yd)
        d[:,:,0] = yd*25
        d[:,:,1] = yd*25
        d[:,:,2] = yd*25
        d = np.uint8(d)
        im = Image.fromarray(d)
        im.save('visualization/stored_examples3/'+str(it)+'_'+str(frame)+'_yygtobjects.png')
        
        single_frame = yhat[-1,0:num_obj]
        single_frame[single_frame<=th] = 0
        new_frame = torch.zeros((num_obj+1,128,128))+0.1
        new_frame[1:] = single_frame
        new_frame = 10 - torch.argmax(new_frame,dim=0)
        new_frame[new_frame==10] = 0
        yhat = new_frame
        yhat = yhat.detach().cpu().numpy()
        yd = yhat
        d = np.zeros((128,128,3))
        yd = np.reshape(yd, (128,128))
        yd = np.double(yd)
        d[:,:,0] = yd*25
        d[:,:,1] = yd*25
        d[:,:,2] = yd*25
        d = np.uint8(d)
        im = Image.fromarray(d)
        im.save('visualization/stored_examples3/'+str(it)+'_'+str(frame)+'_yyobjects.png')

def store_one_samples_with_edge(x,xe,ye,y,yhat,yedge,yhatedge,num_obj,it,frame,task,th=0.5):
    if task == 'VideoObjectSegmentation':
        x = inv_im_trans(x).permute(0,2,3,1)
        xe = inv_im_trans(xe).permute(0,2,3,1)
        cx = x[-1].detach().cpu().numpy()
        yd = cx*255
        d = np.uint8(yd)
        im = Image.fromarray(d)
        im.save('visualization/stored_examples3/'+str(it)+'_'+str(frame)+'_x.png')
        cex = xe[-1].detach().cpu().numpy()
        yd = cex*255
        d = np.uint8(yd)
        im = Image.fromarray(d)
        im.save('visualization/stored_examples3/'+str(it)+'_'+str(frame)+'_xe.png')
        if num_obj == 1:
            obj = 0
        else:
            obj = np.random.randint(0,num_obj-1)
        # cye = ye[-1,num_obj-1,:,:].detach().cpu().numpy()
        # mx = np.max(cye[:])
        # d = np.zeros((224,224,3))
        # yd = cye
        # d[:,:,0] = yd*255
        # d[:,:,1] = yd*255
        # d[:,:,2] = yd*255
        # d = np.uint8(d)
        # im = Image.fromarray(d)
        # im.save('visualization/stored_examples3/'+str(it)+'_'+str(frame)+'_ye.png')

        single_frame = ye[-1,0:num_obj]
        single_frame[single_frame<=th] = 0
        new_frame = torch.zeros((num_obj+1,224,224))+0.1
        new_frame[1:] = single_frame
        new_frame = 10 - torch.argmax(new_frame,dim=0)
        new_frame[new_frame==10] = 0
        ye = new_frame
        ye = ye.detach().cpu().numpy()
        yd = ye
        d = np.zeros((224,224,3))
        yd = np.reshape(yd, (224,224))
        yd = np.double(yd)
        d[:,:,0] = yd*25
        d[:,:,1] = yd*25
        d[:,:,2] = yd*25
        d = np.uint8(d)
        im = Image.fromarray(d)
        im.save('visualization/stored_examples3/'+str(it)+'_'+str(frame)+'_yeobjects.png')

        # cy = y
        # cy = cy[-1,num_obj-1].detach().cpu().numpy()
        # d = np.zeros((128,128,3))
        # yd = cy
        # d[:,:,0] = yd*255
        # d[:,:,1] = yd*255
        # d[:,:,2] = yd*255
        # d = np.uint8(d)
        # im = Image.fromarray(d)
        # im.save('visualization/stored_examples3/'+str(it)+'_'+str(frame)+'_y.png')

        single_frame = y[-1,0:num_obj]
        single_frame[single_frame<=th] = 0
        new_frame = torch.zeros((num_obj+1,128,128))+0.1
        new_frame[1:] = single_frame
        new_frame = 10 - torch.argmax(new_frame,dim=0)
        new_frame[new_frame==10] = 0
        y = new_frame
        y = y.detach().cpu().numpy()
        yd = y
        d = np.zeros((128,128,3))
        yd = np.reshape(yd, (128,128))
        yd = np.double(yd)
        d[:,:,0] = yd*25
        d[:,:,1] = yd*25
        d[:,:,2] = yd*25
        d = np.uint8(d)
        im = Image.fromarray(d)
        im.save('visualization/stored_examples3/'+str(it)+'_'+str(frame)+'_yygtobjects.png')

        # single_frame = yhat[-1,num_obj-1]
        # single_frame[single_frame>0.5] = 1
        # single_frame[single_frame<=0.5] = 0
        # yhatg = single_frame
        # yhatg = yhatg.detach().cpu().numpy()
        # yd = yhatg
        # d = np.zeros((128,128,3))
        # yd = np.reshape(yd, (128,128))
        # yd = np.double(yd)
        # d[:,:,0] = yd*255
        # d[:,:,1] = yd*255
        # d[:,:,2] = yd*255
        # d = np.uint8(d)
        # im = Image.fromarray(d)
        # im.save('visualization/stored_examples3/'+str(it)+'_'+str(frame)+'_yhat.png')
        
        single_frame = yhat[-1,0:num_obj]
        single_frame[single_frame<=th] = 0
        new_frame = torch.zeros((num_obj+1,128,128))+0.1
        new_frame[1:] = single_frame
        new_frame = 10 - torch.argmax(new_frame,dim=0)
        new_frame[new_frame==10] = 0
        yhat = new_frame
        yhat = yhat.detach().cpu().numpy()
        yd = yhat
        d = np.zeros((128,128,3))
        yd = np.reshape(yd, (128,128))
        yd = np.double(yd)
        d[:,:,0] = yd*25
        d[:,:,1] = yd*25
        d[:,:,2] = yd*25
        d = np.uint8(d)
        im = Image.fromarray(d)
        im.save('visualization/stored_examples3/'+str(it)+'_'+str(frame)+'_yyobjects.png')


        single_frame = yedge[-1,0:num_obj]
        single_frame[single_frame<=th] = 0
        new_frame = torch.zeros((num_obj+1,128,128))+0.1
        new_frame[1:] = single_frame
        new_frame = 10 - torch.argmax(new_frame,dim=0)
        new_frame[new_frame==10] = 0
        y = new_frame
        y = y.detach().cpu().numpy()
        yd = y
        d = np.zeros((128,128,3))
        yd = np.reshape(yd, (128,128))
        yd = np.double(yd)
        d[:,:,0] = yd*25
        d[:,:,1] = yd*25
        d[:,:,2] = yd*25
        d = np.uint8(d)
        im = Image.fromarray(d)
        im.save('visualization/stored_examples3/'+str(it)+'_'+str(frame)+'_yedgegtobjects.png')        
        single_frame = yhatedge[-1,0:num_obj]
        single_frame[single_frame<=th] = 0
        new_frame = torch.zeros((num_obj+1,128,128))+0.1
        new_frame[1:] = single_frame
        new_frame = 10 - torch.argmax(new_frame,dim=0)
        new_frame[new_frame==10] = 0
        yhatedge = new_frame
        yhatedge = yhatedge.detach().cpu().numpy()
        yd = yhatedge
        d = np.zeros((128,128,3))
        yd = np.reshape(yd, (128,128))
        yd = np.double(yd)
        d[:,:,0] = yd*25
        d[:,:,1] = yd*25
        d[:,:,2] = yd*25
        d = np.uint8(d)
        im = Image.fromarray(d)
        im.save('visualization/stored_examples3/'+str(it)+'_'+str(frame)+'_yedgeobjects.png')

def store_output(x,xe,ye,y,yhat,yedge,yhatedge,num_obj,video,dataset,frame,task,th):
    if task == 'VideoObjectSegmentation':
        x = inv_im_trans(x).permute(0,2,3,1)
        xe = inv_im_trans(xe).permute(0,2,3,1)
        cx = x[-1].detach().cpu().numpy()
        yd = cx*255
        d = np.uint8(yd)
        im = Image.fromarray(d)
        savepath = 'visualization/'+dataset+'/'+video[0]+'/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        im.save(savepath+str(frame)+'_x.png')
        cex = xe[-1].detach().cpu().numpy()
        yd = cex*255
        d = np.uint8(yd)
        im = Image.fromarray(d)
        im.save(savepath+str(frame)+'_xe.png')

        single_frame = ye[-1,0:num_obj]
        single_frame[single_frame<=th] = 0
        new_frame = torch.zeros((num_obj+1,224,224))+0.1
        new_frame[1:] = single_frame
        new_frame = 10 - torch.argmax(new_frame,dim=0)
        new_frame[new_frame==10] = 0
        ye = new_frame
        ye = ye.detach().cpu().numpy()
        yd = ye
        d = np.zeros((224,224,3))
        yd = np.reshape(yd, (224,224))
        yd = np.double(yd)
        d[:,:,0] = yd*25
        d[:,:,1] = yd*25
        d[:,:,2] = yd*25
        d = np.uint8(d)
        im = Image.fromarray(d)
        im.save(savepath+str(frame)+'_yeobjects.png')

        single_frame = y[-1,0:num_obj]
        single_frame[single_frame<=th] = 0
        new_frame = torch.zeros((num_obj+1,128,128))+0.1
        new_frame[1:] = single_frame
        new_frame = 10 - torch.argmax(new_frame,dim=0)
        new_frame[new_frame==10] = 0
        y = new_frame
        y = y.detach().cpu().numpy()
        yd = y
        d = np.zeros((128,128,3))
        yd = np.reshape(yd, (128,128))
        yd = np.double(yd)
        d[:,:,0] = yd*25
        d[:,:,1] = yd*25
        d[:,:,2] = yd*25
        d = np.uint8(d)
        im = Image.fromarray(d)
        im.save(savepath+str(frame)+'_yygtobjects.png')
        
        single_frame = yhat[-1,0:num_obj]
        single_frame[single_frame<=th] = 0
        new_frame = torch.zeros((num_obj+1,128,128))+0.1
        new_frame[1:] = single_frame
        new_frame = 10 - torch.argmax(new_frame,dim=0)
        new_frame[new_frame==10] = 0
        yhat = new_frame
        yhat = yhat.detach().cpu().numpy()
        yd = yhat
        d = np.zeros((128,128,3))
        yd = np.reshape(yd, (128,128))
        yd = np.double(yd)
        d[:,:,0] = yd*25
        d[:,:,1] = yd*25
        d[:,:,2] = yd*25
        d = np.uint8(d)
        im = Image.fromarray(d)
        im.save(savepath+str(frame)+'_yyobjects.png')



        
    
    
