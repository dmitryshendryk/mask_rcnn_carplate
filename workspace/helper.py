import os 
import sys 
import numpy as np 
import random

ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)



def space_NMS(box_a,box_b):#((x1,y1),(x2,y2))
    width_a=abs(box_a[0][0]-box_a[1][0])
    width_b=abs(box_b[0][0]-box_b[1][0])
    height_a=abs(box_a[0][1]-box_a[1][1])
    height_b=abs(box_b[0][1]-box_b[1][1])
    size_a=width_a*height_a
    size_b=width_b*height_b
    start_x=max(box_a[0][0],box_b[0][0])
    end_x=min(box_a[1][0],box_b[1][0])
    start_y = max(box_a[0][1], box_b[0][1])
    end_y= min(box_a[1][1], box_b[1][1])

    #size_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    center_a=((box_a[0][0]+box_a[1][0])/2,(box_a[0][1]+box_a[1][1])/2)
    center_b=((box_b[0][0]+box_b[1][0])/2,(box_b[0][1]+box_b[1][1])/2)
    if start_x>end_x or start_y>end_y:
        #no overlap
        #print(center_a,center_b)
        return False
    else:

        # overlapsize=((width_a+width_b)-2*(abs(center_a[0]-center_b[0])))*((height_a+height_b)-2*(abs(center_a[1]-center_b[1])))
        # overlapsize=(0.5*(width_a+width_b)-(center_b[0]-center_a[0]))*(0.5*(height_a+height_b)-(center_b[1]-center_a[1]))
        overlapsize=abs(end_x-start_x)*abs(end_y-start_y)
        #print("overlapsize: ", overlapsize, " size_b: ", size_b)
        if overlapsize>=0.5*size_b or overlapsize>=0.5*size_a:

            return  True
        else:
            return False


def aggregate(line,labels,scores,boxs,h_thershold):
    opt_label=[]
    temps=[]
    #print(line,labels,scores,boxs)
    while(len(line)):
        mark = []
        pos=line[0][0]
        label=labels[0]
        score=scores[0]
        box=boxs[0]
        #mark.append(0)

        for i in range(1,len(line),1):
            if not space_NMS(box,boxs[i]):
                mark.append(i)
            elif scores[i]>score:
                    #print("label: ", label)
                    label=labels[i]
                    score=scores[i]
            else:
                #print("label: ",labels[i])
                continue
        newline=[]
        newlabels=[]
        newscores=[]
        newbox=[]
        #print(mark)
        for i in mark:
            newline.append(line[i])
            newlabels.append(labels[i])
            newscores.append(scores[i])
            newbox.append(boxs[i])
        line=newline
        labels=newlabels
        scores=newscores
        boxs=newbox
        temps.append((pos,label))
        #mark.clear()
    temps.sort(key=lambda tu:tu[0])
    for t in temps:
        opt_label.append(t[1])
    return opt_label

def sequence(labels,boxs,scores,v_thershold,h_thershold):
    #first determine wether the car plate is two lines
    is_two_lines=False

    centers=[]
    for box in boxs:
        center=[(box[0][0]+box[1][0])/2.0,(box[0][1]+box[1][1])/2.0]
        centers.append(center)
    # check y
    la=[]
    sc=[]
    lines=[]
    all_boxes=[]
    output=[]
    #print(centers,labels,scores)
    for i in range(len(centers)):
        center=centers[i]
        cur_la=labels[i]
        cur_sc=scores[i]
        cur_box=boxs[i]
        if len(lines)==0: #first
            line_one=[]
            label_one=[]
            sc_one=[]
            box_one=[]

            line_one.append(center)
            lines.append(line_one)

            label_one.append(cur_la)
            la.append(label_one)

            sc_one.append(cur_sc)
            sc.append(sc_one)

            box_one.append(cur_box)
            all_boxes.append(box_one)

        else:
            new_lines=True
            for  i in range(len(lines)):
               is_new_line=True
               for k in range(len(lines[i])):
                    if abs(center[1]-lines[i][k][1])<v_thershold:
                        lines[i].append(center)
                        la[i].append(cur_la)
                        sc[i].append(cur_sc)
                        all_boxes[i].append(cur_box)
                        is_new_line=False
                        break
               if not is_new_line:
                    new_lines=False
                    break
            if new_lines:
                new_line = []
                new_label = []
                new_score = []
                new_box=[]

                new_line.append(center)
                lines.append(new_line)

                new_label.append(cur_la)
                la.append(new_label)

                new_score.append(cur_sc)
                sc.append(new_score)

                new_box.append(cur_box)
                all_boxes.append(new_box)

    #erase the out_lair

    newline=[]
    newscores=[]
    newlabels=[]
    newboxs=[]
    for i in range(len(lines)):
        line=lines[i]
        score=sc[i]
        label=la[i]
        c_box=all_boxes[i]
        #print(c_box)
        if len(line)>=2: #at least 2
            newline.append(line)
            newscores.append(score)
            newlabels.append(label)
            newboxs.append(c_box)
    #determine x

    for i in range(len(newline)):

        line=newline[i]
        label_line=newlabels[i]
        score_line=newscores[i]
        box_line=newboxs[i]
        code=aggregate(line,label_line,score_line,box_line,h_thershold)
        output.append(code)

    if len(output)>2:
        #print(output)
        output=[]

    return output


def get_char_result(image, boxes, masks, class_ids, class_names,
                      scores=None, title="PLC",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None,
                      score_threshold=0.8,show_score=True):

    N = boxes.shape[0]
    print(N)
    global pcl_container
    global  total_count
    if not N:
        print("\n*** No instances to display *** \n")
        return None
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    total_height=0
    for i in range(N):
           y1, x1, y2, x2 = boxes[i]
           height=y2-y1
           total_height+=height
    average_height=total_height/N
    # colors = visualize.get_colors(38)
    # print(colors)
    scoreMin=score_threshold
    ls=[]
    bs=[]
    ss=[]

    for i in range(N):
        if scores[i]>scoreMin:
            print(class_names[class_ids[i]]+' scores:', scores[i])
            color=(1,1,1)
            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            class_id = class_ids[i]
            sc = scores[i]
            label=class_names[class_id]
            box=[(x1,y1),(x2,y2)]
            bs.append(box)
            ls.append(label)
            ss.append(sc)



    v_t=average_height*0.6
    h_t=average_height
    res=sequence(ls,bs,ss,v_t,h_t)
    first=""
    sec=""
    if len(res)>1:
        for c in res[0]:
            first+=c
        for d in res[1]:
            sec+=d
        if len(first)>len(sec):
            temp=first
            first=sec
            sec=temp
    elif len(res)==1:
        if res[0]!="":
            for d in res[0]:
                sec += d
    print(first,sec)

    return first+"\n"+sec
