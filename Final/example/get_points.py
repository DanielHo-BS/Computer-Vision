"""
load csv file
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import copy
from tqdm import tqdm
from skimage.filters import threshold_otsu

def get_point(img,img_Crops,lower,upper,mean,save_dir):
    special_points = []
    #print(len(img_Crops))
    for id,img_crop in (enumerate(img_Crops)):
        thresh_osu = threshold_otsu(img)
        mask = cv2.inRange(img_crop,lower[id],upper)
        img_mask = cv2.bitwise_and(img_crop,img_crop,mask=mask)
        #print(thresh_osu)

        gray = cv2.cvtColor(img_mask,cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(gray,(3,3),0)
        
        #print(canny.shape)
        #_,thresh = cv2.threshold(gray,mean[id],255,0)
        ret,thresh = cv2.threshold(gray,mean[id],255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        _,contours,_= cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        canny = cv2.Canny(gaussian, threshold1 = thresh_osu, threshold2 = 175)
        try :
            os.makedirs("outputs/canny&outputs/"+str(save_dir).strip("\n"))
            
        except FileExistsError:
            pass
        cv2.imwrite("outputs/canny&outputs/"+str(save_dir).strip("\n")+"/canny"+str(id)+".png",canny)

        
        #print("there is",len(contours))

        approx = []
        for cnt in contours:
            approx.append(cv2.approxPolyDP(cnt,3,True))  #corner point

        for j in range(len(approx)):
            c_points = approx[j]
            for i ,_ in enumerate (c_points):
                #cv2.circle(img_crop, (c_points[i][0][0],c_points[i][0][1]), 1, [0,0,255],1)
                #version2 canny mapping
                if canny[c_points[i][0][1]][c_points[i][0][0]] != 0:    
                    cv2.circle(img_crop, (c_points[i][0][0],c_points[i][0][1]), 1, [0,0,255],3)
                    #print(c_points[i][0])
                    special_points.append(c_points[i][0])
                else:
                    continue
        #print(len(special_points))
        try:
            os.makedirs("outputs/getpoints")
        except FileExistsError:
            pass
        cv2.imwrite("outputs//getpoints/"+str(save_dir).strip("\n")+".png",img) 
        cv2.imwrite("outputs/canny&outputs/"+str(save_dir).strip("\n")+"/canny"+str(id)+".png",canny)
    return special_points 
  
def nms(bboxes, scores, iou_thresh):
    """
    :param bboxes: 
    :param scores: 
    :param iou_thresh
    :return:
    """  
    
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    
        
        
    areas = (y2 - y1) * (x2 - x1)

    
    result = []
    index = scores.argsort()[::-1]  
    while index.size > 0:
        
        i = index[0]
        result.append(i) 
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        
        idx = np.where(ious <= iou_thresh)[0]
        index = index[idx + 1]  
    bboxes, scores = bboxes[result], scores[result]
    return bboxes, scores

def main(all_txt,limit):
    with open (all_timestamp,'r') as all_timesf:
        all_lines = all_timesf.readlines()
    all_timesf.close()
    for num_file,all_line in tqdm(enumerate(all_lines)):
        if num_file<limit:
            file_dir = os.path.join("../ITRI_dataset/seq1/dataset/",all_line).strip("\n")
            image = cv2.imread(os.path.join(file_dir,'raw_image.jpg'))
            copy_image = copy.deepcopy(image)
            copy2_image = image.copy()
            csv_path = file_dir+"\detect_road_marker.csv"
            with open (csv_path,"r") as csvfile:
                lines = csv.reader(csvfile)
                bboxes = []
                conf = []
                classes = []
                mono = []
                for line in lines:
                    if float(line[5]) <= 0.2:
                        continue
                    else:

                        line = list(map(float,line))
                        bboxes.append(line[:4])
                        conf.append(line[5])
                        classes.append(int(line[4]))
                        #mono.append(line) 
                        
                csvfile.close
            try :
                bboxes,scores = nms(np.array(bboxes),np.array(conf),0.2)
            except IndexError:
                pass
            mono = list(bboxes)
            
            
            img_crops = []
            mean = []
            for id ,item in enumerate(mono): 
                for i in range(4):
                    item[i] = np.clip(item[i], 1, 9999)
                signs = image[int(item[1])-1:int(item[3])+1,(int(item[0]))-1:int(item[2])+1]
                copy_image = cv2.rectangle(copy_image,(int(item[0]),int(item[1])),(int(item[2]),int(item[3])),(0, 0, 255), 3, cv2.LINE_AA)
                
                cv2.imwrite("test4/box.png",copy_image)
                img_crops.append(signs)
                #print("rrrrrrrrrrrrrrrrrrr",np.std(signs.flatten()))
                
                std =  int(np.std(signs.flatten()))
                
                mean.append(int(signs.flatten().mean())+std)
            lower = []
            if len(mean) >=1:
                for k in range(len(mean)):
                    if mean[k] <250 :
                        lower.append(np.array([mean[k],mean[k],mean[k]]))
                    else:
                        lower.append(np.array([175,175,175]))
            upper = np.array([255,255,255])   
            sp = get_point(image,img_crops,lower,upper,mean,save_dir= all_line) 
            
        else:
            break
        




if __name__ =='__main__':
    dics = ["x1", "y1", "x2", "y2", "class_id", "probability"]
    all_timestamp = "../ITRI_dataset/seq1/all_timestamp.txt"
    main(all_timestamp,9999)



'''
same_class = []
for item,index in classes:
    if same_class == []:
        break
    else:
        same_class.append(item)

        for id in classes[item+1:]:
            if classes[item] == id:
                same_class.append(item)
                classes.remove(item)  
        
        same_class =[]  
'''        
