'''
cd C:/Users/User/Desktop/yolov5
python export.py --weights ./runs/train/exp6/weights/best.pt  --include torchscript onnx

python export.py --weights .\runs\train\exp9\weights\best.pt --imgsz 320 --include torchscript onnx

if detect3 is not working then 

pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 torchtext==0.14.1 fastai==2.7.11
pip install tokenizers
pip install torchdata==0.5.1

'''


################################
# import cv2
# import numpy as np
# from keras.models import load_model
# from numpy import vstack
# import os
import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np 
import torch
import cv2
from os.path import join, isfile
from os import listdir
import numpy as np
import pathlib 
import keras_ocr
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import time


start_time = time.time()
pipeline = keras_ocr.pipeline.Pipeline()

### change me


th = 2
th1 = 10           # always the multiple of th and th1 should be 20 approx. fps
ROOT = 'E:/P1_YOLO/yolov5'
source_path = 'C:/Users/moumi/OneDrive/Desktop/New folder/20250214_112429_1.avi'
dest_path = 'C:/Users/moumi/OneDrive/Desktop/New folder/'

###

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

#F = []
G = []
IMGG = []
NAME = []
# Patha = []
# Ima = []
# Im0sa = []
# Vid_capa = []
# Sa = []
#Path_folder = []
#start_time = time.time()

@smart_inference_mode()
def run(
        weights=ROOT / 'runs/train/stage2_water_weights/weights/best.pt',  # model path or triton URL
        source= source_path ,  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/Wrc_sewer_stage1.yaml',  # dataset.yaml path
        imgsz=(320, 320),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        #project=ROOT / 'runs/detect',  # save results to project/name
        project= dest_path ,  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=True,  # use FP16 half-precision inference
        #half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
   
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    savedirr = str(save_dir)
    #print()
    #AAQQ = savedirr.split("\\")
    #new_path = dest_path + '/' + AAQQ[-1] + '/' 
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 64  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs
    #frame_array = []
    za = 0
    kk_f = 0
    kk_loop = 0 
    seenp = 0
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    #print('Length of full dataset is :', len(dataset))
    
    for path, im, im0s, vid_cap, s in dataset:
        
        kk_f+=1
        if kk_f % th == 0 : 
            kk_loop +=1
            #print('number of frames is :',  kk_f)
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
    
            # Inference
            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)
    
            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        #print('pred=',pred)
            for i, det in enumerate(pred):  # per image
                #if i%5 == 0:
                seenp = seen 
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
    
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
    
                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
    
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(f'{txt_path}.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
    
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            #label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            label = None if hide_labels else (names[c])
                            
                           # ---------------------------------------------------------------------------------------------------
                            G.append(names[c])
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            im0 = annotator.result()
                            #save_path1 = str(Path(save_dir)) + '/' + str(za) + '.jpg'
                            IMGG.append(im0)
                            NAME.append(za)
                            #cv2.imwrite(save_path1, im0)  #### outside mid folder images
                            
                            za += 1
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
    
                  #Stream results
                im0 = annotator.result()
                if view_img:
                    if platform.system() == 'Linux' and p not in windows:
                        windows.append(p)
                        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                        cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond
                    
                #im0 = cv2.resize(im0, [1920,1080])
    ## Stream results
                # im0 = annotator.result()
                # if view_img:
                #     if platform.system() == 'Linux' and p not in windows:
                #         windows.append(p)
                #         cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                #         cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                #     cv2.imshow(str(p), im0)
                #     cv2.waitKey(1)  # 1 millisecond
    
              
    ############        
    
                # Save results (image with detections)
                
                   
                if save_img:
                    if len(det):
                        cv2.imwrite(save_path, im0)
                        #hju = 0
                        #f = [label]
                        #F.append(label)
                        
                    
                    if dataset.mode == 'image':
                        #zqw = 0
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 25, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)
    
    
         
            
            
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    
        # Print results
        if (seen - seenp == 0):
            t = tuple(x.t * 0 * 1E3 for x in dt)  # speeds per image
       
        else:
            t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
            if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
            if update:
                strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    global GG
    GG = np.array(G)
                  
    #np.save(savedirr + '/' +  'FILE_G.npy',GG)
    
    
    
    tpf = float("%.1f" %t[1])
    print('Number of selected frames is:',kk_loop)
    print('Frames per sec: %.1fms' %t[1])
    tot_s = (tpf * kk_loop)/1000
    print ('Total time in sec: %.1fs' %tot_s)
    tot_m = (tpf * kk_loop)/(1000*60)
    print ('Total time in min: %.1fm   ' %tot_m)
    print('final path is :',savedirr)
    file1 = open("pathfile.txt","w")

    file1.write(savedirr)
   
    file1.close()
    
    
    print('folder path is :', savedirr)
    
    path_mid = savedirr + '/mid'
        
    os.mkdir(path_mid)

    #path_prepro = savedirr + '/preprocess' 

    #os.mkdir(path_prepro)
    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/stage2_water_weights/weights/best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default= source_path , help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/Wrc_sewer_stage1.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[320], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    #parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--project', default= dest_path, help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


###############################################################################

Images = np.array(IMGG)
Image_name = np.array(NAME) 

text_as_string = open('pathfile.txt').read()
S = GG
#S = np.load(text_as_string + '/'+ 'FILE_G.npy')
c=[]
p=[]
start = 0
end = 0
k=0
l = 0
F = []

for i in range(0,len(S)):
    c = S[i]
    if (i==0) :
        start = i
        index = c
    elif (p == c):
        index = c
        end = end+1
        if (i == (len(S)-1)):
            index = c

            end = i
            f = [index,start,end]
            F.append(f)

    else:
        f = [index,start,end]
        start = end+1
        F.append(f)
    p = c
        
F1 = np.array(F)
#print(F1)
if len(F1)==1 :
  Z1 = F1

else:   
    G = []
    for j in range(0,len(F1)):
        A = F1[j][0]
        start = F1[j][1]
        end = F1[j][2]
        if j!= (len(F1)-1):
            B = F1[j+1,0]
            if A == B:
                endd = F1[j+1,2]
                g = [A,start,endd]
            else:
                g = [A,start, end]
        else:
            g = [A,start,end]
                
        G.append(g)
                
        
    G1 = np.array(G)        
    G2 = np.copy(G1)

    for k in range(len(G2)):
        A   = G2[k][0]
        af  = int(G2[k][1])
        ase = int(G2[k][2])
        #print(len(G1))
        if af < ase:
            #if ((ase- af)>5):
            if ((ase- af)>th1):    
              if k != (len(G2)-1):
                B = G2[k+1][0]
                if A == B:
                    G1[k+1,:] = 0
            else:
                G1[k,:] = 0
        else:
            G1[k,:] = 0
      
    G12 = np.array(G1)
    G21 = np.copy(G12)
    
    Z = []
    
    for m in range(len(G21)):
        q1 = np.array(G12[m,1],dtype = 'int32')
        q2 = np.array(G12[m,2],dtype = 'int32')
        q = G12[m][:]
        if (np.sum(q1+q2) != 0):
            Z.append(q)
    
    
    Z1 = np.array(Z)
    print(Z1) 
    Z2 = Z1[:,0]
    filename = 'E:/YOLO_model_testing/EN_sewer_stage2/Euro_test_stage2/sample_report_format/New folder (2)/defect.txt'
    with open(filename,'w') as file:
        file.write(str(Z2))
        
    

    # def save_variable(variable, file_name):
    #     with open(file_name, 'w') as file:
    #         file.write(str(variable))
     

    # save_variable(Z2, "defect.txt")
    
    
   # np.save('E:/YOLO_model_testing/EN_sewer_stage2/Euro_test_stage2/sample_report_format/New folder (2)/Defect_list.npy',Z1[0])
    #np.save('Defect_list.npy',Z1)
     

# import cv2
# from os.path import join, isfile
# from os import listdir
# #from matplotlib import pyplot as plt
# import numpy as np
dist_m = []
mypath = text_as_string 
path_mid = text_as_string + '/' + 'mid/'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f))]

for i in range(len(Z1)):
    sf = np.array(Z1[i][1],dtype = 'int32')
    se = np.array(Z1[i][2],dtype = 'int32')
    smm = np.array(((sf+se)/2),dtype = 'int32')
    sm = np.array(np.ceil(smm),dtype = 'int32')
    '''print('sf = ',sf)
    print('se = ',se)
    print('sm = ',sm)'''
    sm1 =  str(sm) + '.jpg'
    I_m = Images[sm]
    
    #I_m = cv2.imread(join(mypath, sm1))
    filemid   = join(path_mid, sm1)
    cv2.imwrite(filemid,I_m) 
    
    image = keras_ocr.tools.read(filemid)

#image_path = '31.jpg'
#image = keras_ocr.tools.read(image_path)

# Detect text in the image
    prediction_groups = pipeline.recognize([image])

# Define the specific location (bounding box)
    x1, y1, x2, y2 = 820, 4, 1100, 40  # Example coordinates


# Extract text from the specified location
    for prediction in prediction_groups[0]:
        box = prediction[1]
        if (box[0][0] >= x1 and box[0][1] >= y1 and box[2][0] <= x2 and box[2][1] <= y2):
            #print(prediction[0])
            dist_m.append(prediction[0])
            

Dist_mm = np.array(dist_m)    

filename_dist = 'E:/YOLO_model_testing/EN_sewer_stage2/Euro_test_stage2/sample_report_format/New folder (2)/distance.txt'   

with open(filename_dist, 'w') as filed:
    filed.write(str(Dist_mm))
    
    

# from os import listdir
# from os.path import isfile, join

# def histo_eq_drak_bright(test_image):
#     img_yuv = cv2.cvtColor(test_image, cv2.COLOR_BGR2YUV)
#     img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
#     img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
#     return img_output
    
# def image_enhan_deblurr(test_image): 
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#     sharpened_image = cv2.filter2D(test_image, -1, kernel)
#     return sharpened_image

# s_path = path_mid 
# d_path = text_as_string + '/' + 'preprocess/'
# S = listdir(s_path)
# #i=0

# for i in range(0,len(S)):
#     test_imag = cv2.imread(s_path + S[i])
#     fin_img2 = histo_eq_drak_bright(test_imag)
#     fin_img3 = image_enhan_deblurr(fin_img2)
#     j = i+1
#     cv2.imwrite(d_path + S[i],fin_img3) 

print("--- %s seconds ---" % (time.time() - start_time))
###############################################################################

# In[2] how to seperate string inside bracket

'''
 Explanation of the Regex
\(: Matches the opening parenthesis.
 (.*?): Captures any characters (non-greedy) until the first comma.
 ,\s*: Matches a comma followed by any whitespace (if present).
 (.*?): Captures any characters (non-greedy) after the comma until the closing parenthesis.
 \): Matches the closing parenthesis.
'''

#ww = Z2[0]
import re
defect_code = []
grade = []

#s= "mystring name is (kaif,3)"

# cheese = (re.findall('\((.*?),\s*(.*?)\)',ww))

# print(cheese)


for i in range(0,len(Z2)):
    dew = Z2[i]
    cheese = (re.findall('\((.*?),\s*(.*?)\)',dew))
    defect_code.append(cheese[0][0])
    grade.append(cheese[0][1])
    

Defe = np.array(defect_code)
gra = np.array(grade)

filename_defect = 'E:/YOLO_model_testing/EN_sewer_stage2/Euro_test_stage2/sample_report_format/New folder (2)/defect_code.txt'

with open(filename_defect,'w') as filed:
    filed.write(str(Defe))
    
filename_grade = 'E:/YOLO_model_testing/EN_sewer_stage2/Euro_test_stage2/sample_report_format/New folder (2)/defect_grade.txt'

with open(filename_grade, 'w') as fileg:
    fileg.write(str(gra))

