import argparse
import logging
import time

import pyrealsense2 as rs
import cv2
import numpy as np
import math

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
import common

#====================
from socket import socket, AF_INET, SOCK_STREAM
import threading
import time
from tkinter import *
import tkinter as tk
import os
from PIL import ImageTk as itk, Image
import json, types,string

# ==========================
# 以下為一些Global函式
# ==========================
# Global var
chating_room=''
chat_text=''
current_text_index=-1
is_tracking = False
logger=None
window=None

class JointDepthData:
    def __init__(self,number,x,y,depth):
        self.jn=number
        self.x = x
        self.y = y
        self.dp=depth
        
class HumanData:
    def __init__(self,joints,joint_images):
        self.joint_list=joints
        self.image_list=joint_images
#   更新聊天室的訊息    
def update_chat_text():
    global current_text_index,chating_room,chat_text
    while current_text_index<len(chating_room.chat_text_list)-1:
        current_text_index+=1
        the_string = chating_room.chat_text_list[current_text_index][1]+"說:"+chating_room.chat_text_list[current_text_index][2]+'\n'
        chat_text.configure(state="normal")
        chat_text.insert(INSERT, the_string)
        chat_text.configure(state="disabled")
    chat_text.see("end")
        
#   本地端新增文字到聊天室的訊息上
def showText(chating_room,speaker,text):
    chating_room.chat_text_list.append(['0',speaker,text])
    update_chat_text()
    
#   取得本地端檔案路徑
def getFilePath(filename):
    script_dir = os.getcwd() #<-- absolute dir the script is in
    return os.path.join(script_dir, filename)

def match_img(image,Target): 
    import cv2 
    import numpy as np 
    img_gray = image.astype(np.uint8)
    # img_gray= cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) 
    template = Target.astype(np.uint8)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED) 
    threshold = 0.8 
    loc = np.where( res >= threshold) 
        
    return loc
 

def doCNNTracking(args):    
    global is_tracking,logger

    fps_time = 0

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    logger.debug('cam read+')
    pipeline = rs.pipeline()
    pipeline.start()
    
    frames = pipeline.wait_for_frames()
    #get depth影像
    depth = frames.get_depth_frame() 
    depth_image_data = depth.as_frame().get_data()
    depth_image = np.asanyarray(depth_image_data)
    
    
    
    logger.info('cam depth image=%dx%d' % (depth_image.shape[1], depth_image.shape[0])) 
    logger.info('camera ready') 
    
    
    #計算depth影像對應至rgb影像的clip
    clip_box = [100,-100,290,-300]
    
    human_list = []
    while (True):
        if(is_tracking):
            
            fps_time = time.time()
            frames = pipeline.wait_for_frames()
            #get rgb影像
            image_frame = frames.get_color_frame()
            image_data = image_frame.as_frame().get_data()
            image = np.asanyarray(image_data)
            
            #change bgr 2 rgb
            image = np.array(image[...,::-1])
            origen_image = image

            #get depth影像
            depth = frames.get_depth_frame() 
            depth_image_data = depth.as_frame().get_data()
            depth_image = np.asanyarray(depth_image_data)
            depth_image = depth_image[(int)(clip_box[0]):(int)(clip_box[1]),(int)(clip_box[2]):(int)(clip_box[3])]
            depth_image = cv2.resize(depth_image, (640, 480), interpolation=cv2.INTER_CUBIC)
            depth_image=depth_image/30
            depth_image.astype(np.uint8)
            
            #深度去背的遮罩
            thresh=cv2.inRange(depth_image,20,160)

            #去背的遮罩做影像處理
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 1))  
            eroded = cv2.erode(thresh,kernel)
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))  
            dilated = cv2.dilate(eroded,kernel2)
            
            #亮度遮罩
            bright_mask = np.zeros(image.shape);
            bright_mask.fill(200)
            bright_mask = bright_mask.astype(np.uint8);
            bright_mask = cv2.bitwise_and(bright_mask, bright_mask, mask=dilated)
            
            #rgb影像亮度處理
#             image = cv2.bitwise_and(image, image, mask=dilated)
            
            image = image.astype(int)+200-bright_mask.astype(int);
            image = np.clip(image, 0, 255)
            image = image.astype(np.uint8);
            
            image=origen_image
            
            #影像邊緣檢測
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(image_gray, (3, 3), 0)
            canny = cv2.Canny(blurred, 50, 50)
            canny_blurred = cv2.GaussianBlur(canny, (13, 13), 0)
            cv2.imshow('test', canny_blurred)
            
            #tfpose image 縮放
            if args.zoom < 1.0:
                canvas = np.zeros_like(image)
                img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
                dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
                dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
                canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
                image = canvas
            elif args.zoom > 1.0:
                img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
                dx = (img_scaled.shape[1] - image.shape[1]) // 2
                dy = (img_scaled.shape[0] - image.shape[0]) // 2
                image = img_scaled[dy:image.shape[0], dx:image.shape[1]]

            #tfpose 計算
            humans = e.inference(image)
            
            # #得到joint
            # jdata = TfPoseEstimator.get_json_data(image.shape[0],image.shape[1],humans)
            # if(len(jdata)>2):
            #     try:
            #         #傳送Position資料至SERVER
            #         chating_room.sendTrackingData(jdata,'track')
            #     except:
            #         print("Cannot send data to server.")
           
            
            #去背後深度影像
            depth_masked = cv2.bitwise_and(depth_image, depth_image, mask=dilated)
            new_human_list = []
            for human in humans:
                #計算深度資料
                depthDatas=[None] * 20
                image_h, image_w = image.shape[:2]
                
                               
                # get joint data with depth
                for i in range(common.CocoPart.Background.value):
                    if i not in human.body_parts.keys():
                        continue
                    body_part = human.body_parts[i]
                    y= int(body_part.y * image_h+ 0.5)
                    x = int(body_part.x * image_w + 0.5)
                    s=5;
                    mat = depth_masked[y-s if(y-s>=0) else 0:y+s if(y+s<=479) else 479,x-s if(x-s>=0) else 0:x+s if (x+s<=639) else 639]

                    count=0;
                    sum_depth=0;

                    for j in range (mat.shape[0]):
                        for k in range (mat.shape[1]):     
                            if mat[j,k]!=0:
                                sum_depth+=mat[j,k]
                                count+=1
                                
                    if(count>0):
                        depth=sum_depth/count
                    else:
                        depth=0
                    
                    mat = depth_masked[y-s if(y-s>=0) else 0:y+s if(y+s<=479) else 479,x-s if(x-s>=0) else 0:x+s if (x+s<=639) else 639]
                    
                    try:
                        depthDatas[i] =  (JointDepthData(i,x,y,depth))
                    except:
                        print("err:"+str(x)+" "+str(y)+" "+str(body_part.x )+" "+str(body_part.y ))
                
                # 後處理                
                jn0=np.zeros((20,3))
                
                for j in depthDatas:
                    if(j!=None):
                        jn0[j.jn]=np.array([1,j.x,j.y])
                jn0 = jn0.astype(int)
                jn00 = jn0.copy()
                
                old_images=None
                if(len(human_list)>0):
                # 找與之前最相似的Human Data  
                    most_simular_value = 9999999
                    most_simular_human = ''
                    jn1=np.zeros((20,3))
                    
                    for human_data in human_list:                        
                        for j in human_data.joint_list:
                            if(j != None):
                                jn1[j.jn]=np.array([1,j.x,j.y])
                        jn1 = jn1.astype(int)

                        different_value=0
                        match_count=0
                        for i in range (18) :
                            if(jn0[i,0]*jn1[i,0]!=0):
                                different_value+= np.linalg.norm(jn0[i]-jn1[i])
                                match_count+=1
                        if(different_value/match_count<most_simular_value):
                            most_simular_human = human_data
                
                    old_images = most_simular_human.image_list
                    
                    if(old_images==None):
                        old_images = [np.zeos((80,80))]*18
                    w = 160
                    
                    
#                     for joint_i in range (18):
#                         smallest_diff = 9999999
#                         if(jn0[joint_i,0]==int(1) and jn1[joint_i,0]==int(1) and old_images[joint_i].shape[0]>0 and old_images[joint_i].shape[1]>0):
#                             new_center=np.zeros(2)
#                             for i in range (-20,20,10):
#                                 for j in range (-20,20,10):
#                                     center = np.array([jn0[joint_i,1]+i,jn0[joint_i,2]+j])
                                    
#                                     mat0 = canny_blurred[center[0]-w:center[0]+w,center[1]-w:center[1]+w]
#                                     if(mat0.shape[0]!=2*w or mat0.shape[1]!=2*w or old_images[joint_i].shape[0]!=2*w or old_images[joint_i].shape[1]!=2*w):
#                                         continue
#                                     try:
#                                         mat1 = mat0-old_images[joint_i]
#                                     except:
#                                         print(mat0.shape)
#                                     mat2 = np.exp2(mat1)
#                                     diff = np.sum(mat2)
#                                     if(diff<smallest_diff):
#                                         smallest_diff=diff
#                                         new_images[joint_i] = canny_blurred[center[0]-w:center[0]+w,center[1]-w:center[1]+w]
#                                         new_center=center
#                             if(smallest_diff<9999999):
#                                 jn0[joint_i,1] = int(new_center[0]+jn0[joint_i,1])/2
#                                 jn0[joint_i,2] = int(new_center[1]+jn1[joint_i,1])/2
                    temp = np.copy(image)
                    
                    for joint_i in range(18):
                        center = np.array([jn0[joint_i,1],jn0[joint_i,2]])
                        mat0 = image_gray[center[1]-w if(center[1]-w>=0) else 0:center[1]+w if(center[1]+w<480) else 479,center[0]-w if(center[0]-w>=0) else 0:center[0]+w if(center[0]+w<640) else 639]
                        mat1 = old_images[joint_i]
                        if(mat0.shape[0]>mat1.shape[0]and mat0.shape[1]>mat1.shape[1] and mat0.shape[0]>0 and mat0.shape[1]>0 and mat1.shape[0]>0 and mat1.shape[1]>0):
                            loc = match_img(mat0,mat1)
                            lt = (center[0]-w if(center[0]-w>=0) else 0,center[1]-w if(center[1]-w>=0) else 0)
                            
                            
                            if(len(loc[1])>0):                                 
                                c = (int(sum(loc[1])/len(loc[1])),int(sum(loc[0])/len(loc[0])))
                                cv2.rectangle(temp,(c[0]+lt[0],c[1]+lt[1]),(c[0]+lt[0]+40,c[1]+lt[1]+40), common.CocoColors[joint_i], 2)                             
                            # for pt in : 
                            #     c = pt-np.array((center[1]-w if(center[1]-w>=0) else 0,center[0]-w if(center[0]-w>=0) else 0))+np.array((center[1],center[0]))
                            #     c2 = (c[0],c[1])
                                # cv2.rectangle(temp,c2,(c[0]+40,c[1]+40),(255,255,255), 2) 
                                
                    
                    cv2.imshow('test3',  temp )
                          
 
                
                w = 40
                new_depthDatas=[None]*20
                new_images = [np.zeros((80,80))]*20
                for joint_i in range (18):
                    if(jn0[joint_i,0]==1):
                        new_depthDatas[joint_i]=(JointDepthData(joint_i,jn0[joint_i,1],jn0[joint_i,2],depthDatas[joint_i].dp))
                        m=np.copy(image_gray[jn0[joint_i,2]-w:jn0[joint_i,2]+w,jn0[joint_i,1]-w:jn0[joint_i,1]+w])

                        if(old_images==None and m.shape[0]==2*w and m.shape[1]==2*w):
                            cv2.circle(image,(jn0[joint_i,1],jn0[joint_i,2]), w, (0,0,255), -1)
                            new_images[joint_i]=m
                        elif(old_images==None and (m.shape[0]<2*w or m.shape[1]<2*w)):
                            pass
                        elif(m.shape[0]==2*w and m.shape[1]==2*w):
                            cv2.circle(image,(jn0[joint_i,1],jn0[joint_i,2]),w, (0,255,0), -1)
                            new_images[joint_i]=old_images[joint_i]*0.9+m*0.1
                        else:
                            cv2.circle(image,(jn0[joint_i,1],jn0[joint_i,2]), w, (255,0,0), -1)
                            new_images[joint_i]=old_images[joint_i]
                        
                        if(joint_i==1 and m.shape[0]==2*w and m.shape[1]==2*w):
                            cv2.imshow('test2', m)
                            cv2.circle(image,(jn00[joint_i,1],jn00[joint_i,2]), w, (255,255,0), -1)
                    
                        # new_images.append(depth_masked[jn0[joint_i,1]-5:jn0[joint_i,1]+5,jn0[joint_i,2]-5:jn0[joint_i,2]+5])
                        
                
                
                new_human = HumanData(new_depthDatas,new_images)
                
                new_human_list.append(new_human)
                
#                 depth_jdata=json.dumps(new_depthDatas)
#                 if(len(depth_jdata)>2):
#                     try:
#                         #傳送Depth資料至SERVER
#                         print(depth_jdata)
#                         chating_room.sendTrackingData(depth_jdata,'track_depth')

#                     except:
#                         print("Cannot send depth data to server.")
            
            human_list = new_human_list           
            depth_image =  cv2.applyColorMap(cv2.convertScaleAbs(depth_image/25), cv2.COLORMAP_JET)
#             cv2.circle(image,(320,240), 5, (255,0,0), -1)

            
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

            cv2.putText(image,"FPS: %f" % (1.0 / (time.time() - fps_time)),(10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
            
            cv2.imshow('tf-pose-estimation result', image)
           
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


    cv2.destroyAllWindows()

	


# ==========================
# client端的聊天室物件
# ==========================
class SocketClient(socket):
    #儲存server的socket
    serverip=''
    serverport=''
    serversocket=''
    username=''
    #接收server端資料的thread    
    recv_thread=''
    #儲存聊天室的訊息
    chat_text_list=[]
    
    #     =====================
    #     HERE IS SERVER INIT
    #     =====================    
    def __init__(self,ip_,port_,username):
        self.serversocket= socket(AF_INET, SOCK_STREAM)
        self.serverip = ip_
        self.serverport = port_
        self.init_UI(username)
        self.username=username
    # ==========================
    # 以下程式碼負責聊天室UI顯示
    # ==========================
    def init_UI(self,username):
        global window,chat_text
        def setShape(widget,dict_,anchor_):
            widget.place(x=dict_['x'], y=dict_['y'], width=dict_['width'], height=dict_['height'], anchor=anchor_)

    #     一些參數設定
        TEXT_SIZE=15

        CHAT_TEXTBOX_SHAPE={'x':10,'y':10,'width':440,'height':300}
        CHAT_TEXTBOX_MAX_LINE=5
        CHAT_TEXTB0X_SCROLLBAR_SHAPE={'x':460,'y':10,'width':30,'height':300}

        INPUTBOX_SHAPE={'x':10,'y':320,'width':320,'height':100}
        INPUTBOX_SCROLLBAR_SHAPE={'x':330,'y':320,'width':30,'height':100}

        SEND_MSG_BTN_SHAPE={'x':380,'y':320,'width':100,'height':100}
    #     聊天室訊息視窗中
        
        chat_text_scrollbar = Scrollbar(window) 

        chat_text = tk.Text(window,yscrollcommand=chat_text_scrollbar.set)
        chat_text.config(font=("Courier", TEXT_SIZE))
        chat_text.place(x=CHAT_TEXTBOX_SHAPE['x'], y=CHAT_TEXTBOX_SHAPE['y'], width=CHAT_TEXTBOX_SHAPE['width'], height=CHAT_TEXTBOX_SHAPE['height'], anchor='nw')

        chat_text_scrollbar.config(command=chat_text.yview)  
        chat_text_scrollbar.place(x=CHAT_TEXTB0X_SCROLLBAR_SHAPE['x'], y=CHAT_TEXTB0X_SCROLLBAR_SHAPE['y'], width=CHAT_TEXTB0X_SCROLLBAR_SHAPE['width'],height=CHAT_TEXTB0X_SCROLLBAR_SHAPE['height'], anchor='nw')
    #     文字輸入欄位視窗
        inputbox_scrollbar = Scrollbar(window)  

        inputbox = tk.Text(window,  yscrollcommand=inputbox_scrollbar.set)
        inputbox.config(font=("Courier", TEXT_SIZE))
        inputbox.place(x=INPUTBOX_SHAPE['x'], y=INPUTBOX_SHAPE['y'], width=INPUTBOX_SHAPE['width'], height=INPUTBOX_SHAPE['height'], anchor='nw')

        inputbox_scrollbar.config(command=inputbox.yview) 
        inputbox_scrollbar.place(x=INPUTBOX_SCROLLBAR_SHAPE['x'], y=INPUTBOX_SCROLLBAR_SHAPE['y'], width=INPUTBOX_SCROLLBAR_SHAPE['width'],height=INPUTBOX_SCROLLBAR_SHAPE['height'], anchor='nw')


   


    # ================================
    # 以下程式碼負責UI(用於伺服器連接)的設定
    # =================================
    #     一些參數設定
        SERVER_CONN_BTN_SHAPE={'x':410,'y':430,'width':80,'height':70}
        SERVER_IP_ENTRY_SHAPE={'x':210,'y':430,'width':180,'height':30}
        SERVER_PORT_ENTRY_SHAPE={'x':210,'y':470,'width':180,'height':30}
        TXT_SERVER_IP_ENTRY_SHAPE={'x':10,'y':430,'width':180,'height':30}
        TXT_SERVER_PORT_ENTRY_SHAPE={'x':10,'y':470,'width':180,'height':30} 

        USERNAME_SHAPE={'x':210,'y':510,'width':180,'height':30}
        TXT_USERNAME_SHAPE={'x':10,'y':510,'width':180,'height':30}     
    #   SERVER的IP設定
        txt_ip = StringVar(window, value='Server IP:')
        txt_server_ip_entry= Entry(window,textvariable=txt_ip)
        txt_server_ip_entry.config(font=("Courier", TEXT_SIZE),state="disabled")
        txt_server_ip_entry.place(x=TXT_SERVER_IP_ENTRY_SHAPE['x'], y=TXT_SERVER_IP_ENTRY_SHAPE['y'], width=TXT_SERVER_IP_ENTRY_SHAPE['width'], height=TXT_SERVER_IP_ENTRY_SHAPE['height'], anchor='nw')

        txt_ip = StringVar(window, value=self.serverip)
        server_ip_entry = Entry(window,textvariable=txt_ip)
        server_ip_entry.place(x=SERVER_IP_ENTRY_SHAPE['x'], y=SERVER_IP_ENTRY_SHAPE['y'], width=SERVER_IP_ENTRY_SHAPE['width'], height=SERVER_IP_ENTRY_SHAPE['height'], anchor='nw')

    #   SERVER的Port設定 
        txt_port = StringVar(window, value='Server Port:')
        txt_server_port_entry = Entry(window, textvariable=txt_port)
        txt_server_port_entry.config(font=("Courier", TEXT_SIZE),state="disabled")
        txt_server_port_entry.place(x=TXT_SERVER_PORT_ENTRY_SHAPE['x'], y=TXT_SERVER_PORT_ENTRY_SHAPE['y'], width=TXT_SERVER_PORT_ENTRY_SHAPE['width'], height=TXT_SERVER_PORT_ENTRY_SHAPE['height'], anchor='nw')

        txt_port = StringVar(window, value=self.serverport)
        server_port_entry = Entry(window,textvariable=txt_port)
        setShape(server_port_entry,SERVER_PORT_ENTRY_SHAPE,'nw')

    #   使用者的名字設定
        txt_user = StringVar(window, value='User Name:')
        txt_username_entry = Entry(window, textvariable=txt_user)
        txt_username_entry.config(font=("Courier", TEXT_SIZE),state="disabled")
        setShape(txt_username_entry,TXT_USERNAME_SHAPE,'nw')

        txt_user = StringVar(window, value=username)
        username_entry = Entry(window,textvariable=txt_user)
        setShape(username_entry,USERNAME_SHAPE,'nw')

    #   與SERVER連接的按鈕
        server_conn_btn = tk.Button(window, text='連線' ,command = lambda: chating_room.run(server_ip_entry.get(),server_port_entry.get(),username_entry.get()))
        setShape(server_conn_btn,SERVER_CONN_BTN_SHAPE,'nw')
        
#   視窗中的傳送訊息按鈕
        send_msg_btn = tk.Button(window, text='送出' ,command =lambda:chating_room.sendButtonClick(inputbox))
        setShape(send_msg_btn,SEND_MSG_BTN_SHAPE,'nw')

    # ================================
    # 以下程式碼負責程式特殊功能...
    # =================================
    #     一些參數設定
        GETTIME_BTN_SHAPE={'x':10,'y':560,'width':100,'height':30}
        RUNTRACKING_BTN_SHAPE={'x':130,'y':560,'width':100,'height':30}
        STOPTRACKING_BTN_SHAPE={'x':250,'y':560,'width':100,'height':30}
    #   取得時間
        gettime_btn = tk.Button(window, text='現在幾點?' ,command = lambda: chating_room.getTime())
        setShape(gettime_btn,GETTIME_BTN_SHAPE,'nw')

        def startTracking():
            global is_tracking,logger
            logger.info('Start Tracking')
            is_tracking = True

        def stopTracking():
            global is_tracking,logger
            logger.info('Stop Tracking')
            is_tracking = False

    #   執行追蹤的按鈕
        runTracking_btn = tk.Button(window, text='開始追蹤' ,command = lambda: startTracking())
        setShape(runTracking_btn,RUNTRACKING_BTN_SHAPE,'nw')

    #   停止追蹤的按鈕
        runTracking_btn = tk.Button(window, text='停止追蹤' ,command = lambda: stopTracking())
        setShape(runTracking_btn,STOPTRACKING_BTN_SHAPE,'nw')

#     負責接收server端資料
#     會用threading呼叫執行持續監聽的動作
#     目前收到資料時只將資料顯示出來
    def recieve(self):
        global logger
        while 1:
            data = self.serversocket.recv(8192)
            try:
                jdata = json.loads(data)
            except:
                print("get error json")
                print(data)
                continue
            logger.info('got data=>' + jdata['cmd'])
            recv_cmd = jdata['cmd']
            
            if recv_cmd == 'disconnect':
                break
            elif recv_cmd == 'say':            
                self.chat_text_list.append([0,jdata['name'],jdata['data']])
                update_chat_text()
        self.disconnect(client)

#     程式剛開始運行時會執行run()
#     進行與SERVER連接的動作
    def run(self,ip,port,username):
        print(self.username+" "+username)
        self.serverip = ip
        self.serverport = port
        self.username = username
        
        try:
            conn_string="與伺服器(%s:%s)進行連接" % (self.serverip,self.serverport,)
            showText(self,"Rem",conn_string)
            self.serversocket.connect((self.serverip, int(self.serverport)))
            self.onconnect()
            threading.Thread(target=self.recieve).start()
        except Exception as ex:
            print (ex)
            self.onconnectfailure()
        finally:
            pass
    # 送出文字到SERVER
    def sendButtonClick(self,inputbox):
        try:
            msg = {'cmd':'say','name':self.username, 'data':inputbox.get("1.0",END)[:-1]}
            jmsg = json.dumps(msg)
            chating_room.sendString(jmsg)
            inputbox.delete('1.0', END)
        except Exception as ex:
            print (ex)
            showText(chating_room,"Rem","無法送出文字到SERVER QQ")
        finally:
            pass

#     sendString()負責傳送一段句子到server
    def sendString(self,string):
        self.serversocket.send(str.encode(string)) 
#     sendString()負責傳送一段句子到server
    def sendTrackingData(self,string,cmd):
        msg = {'cmd':cmd,'name':self.username, 'data':string}
        jmsg = json.dumps(msg)
        # print(jmsg)
        self.sendString(jmsg) 

#     getTime()負責從server取得當前時間
    def getTime(self):
        msg = {'cmd':'what time','name':'', 'data':''}
        jmsg = json.dumps(msg)
        self.serversocket.send(str.encode(jmsg))

#     disconnect()負責從server取得當前時間
    def disconnect(self):
        self.onclose(client)
        #Closing connection with client
        self.serversocket.close()
        #Closing thread
        if(recv_thread!=None):
            recv_thread.exit()
    def reconnect(self):
        self.serversocket.close()
        #Closing thread
        if(recv_thread!=None):
            recv_thread.exit()
            
    # ===================
    # SOME EVENT FUNCTION
    # ===================
    def onconnectfailure(self):
        showText(self,"Rem","目標電腦拒絕連線")
    def onconnect(self):
        showText(self,"Rem","成功與伺服器端連接")
        
    def onmessage(self, message):
        print ("Got message ",message)
        
    def onclose(self, client):
        print ("Disconnected")



# ==========================
# 以下為主程式碼
# ==========================
def main():
    global chating_room,window,chat_text,logger,cnn_args
    current_text_index=-1

#    引入參數   
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    cnn_args = parser.parse_args()
    
#    設定logger    
    logger = logging.getLogger('TfPoseEstimator-WebCam')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

#  執行追蹤threading 
    th = threading.Thread(target=doCNNTracking, args = (cnn_args,))
    th.start()    

#    產生程式視窗並初始化
    window = tk.Tk()
    window.title('RemChat')
    window.geometry('1200x600')    

#   帳號設定
    default_username="track00"
    default_IP="127.0.0.1"
    default_port="5566"
    
#    產生一個聊天室
    chating_room = SocketClient(default_IP,default_port,default_username)

    showText(chating_room,"Rem","歡迎使用聊天室~")
    showText(chating_room,"Rem","請輸入伺服器ID和PORT進行連線~")
    window.mainloop()
	
if __name__ == "__main__":
    main()