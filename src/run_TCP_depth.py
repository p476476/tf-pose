import argparse
import logging
import time

import pyrealsense2 as rs
import cv2
import numpy as np

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
            thresh=cv2.inRange(depth_image,20,200)

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
            
#             #得到joint
#             jdata = TfPoseEstimator.get_json_data(image.shape[0],image.shape[1],humans)
#             if(len(jdata)>2):
#                 try:
#                     #傳送Position資料至SERVER
#                     chating_room.sendTrackingData(jdata,'track')
#                 except:
#                     print("Cannot send data to server.")
           
            
            #去背後深度影像
            depth_masked = cv2.bitwise_and(depth_image, depth_image, mask=dilated)
            
            human_json_datas = []
            for human in humans:
                #計算深度資料
                depthDatas=[]
                image_h, image_w = image.shape[:2]
                # get point
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
                        
                    try:
                        depthDatas.append(JointDepthData(i,x,y,depth).__dict__)
                    except:
                        print("err:"+str(x)+" "+str(y)+" "+str(body_part.x )+" "+str(body_part.y ))
                    

                human_json_datas.append(json.dumps(depthDatas))
            json_data = json.dumps(human_json_datas)
            if(len(json_data)>2):
                try:
                    #傳送Depth資料至SERVER
                    chating_room.sendTrackingData(json_data,'track_depth')
                except:
                    print("Cannot send depth data to server.")
                        
            depth_image =  cv2.applyColorMap(cv2.convertScaleAbs(depth_image/25), cv2.COLORMAP_JET)
            cv2.circle(image,(320,240), 5, (255,255,255), -1)
            cv2.circle(image,(304,480-98), 5, (0,0,255), -1)
            cv2.circle(image,(377,480-197), 5, (0,160,255), -1)
            cv2.circle(image,(106,480-49), 5, (0,255,255), -1)
            cv2.circle(image,(460,480-136), 5, (0,255,0), -1)
            cv2.circle(image,(481,480-134), 5, (255,0,0), -1)
            cv2.circle(image,(85,480-143), 5, (255,160,0), -1)
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