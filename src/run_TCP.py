import argparse
import logging
import time

import cv2
import numpy as np

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

#====================
from socket import socket, AF_INET, SOCK_STREAM
import threading
import time
from tkinter import *
import tkinter as tk
import os
from PIL import ImageTk as itk, Image
import json, types,string
import wx
import wx.xrc

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
close_recv=False


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
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()

    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    

    while (True):
        if(is_tracking):
            ret_val, image = cam.read()


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


            humans = e.inference(image)
            jdata = TfPoseEstimator.get_json_data(image.shape[0],image.shape[1],humans)

            try:
            #傳送資料至SERVER
                if(len(jdata)>2):
                    chating_room.sendTrackingData(jdata)
            except:
                print("Cannot send data to server.")
                


            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

            logger.debug('show+')
            
            cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()
            time.sleep(0.1)

            if cv2.waitKey(1) == 27:
                break
    cv2.destroyAllWindows()
    cam.release()

	


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
        RECONN_BTN_SHAPE={'x':370,'y':560,'width':100,'height':30}
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
            
        def reconnecting(chating_room):
            global is_tracking,logger
            chating_room.reconnect();
            

    #   執行追蹤的按鈕
        runTracking_btn = tk.Button(window, text='開始追蹤' ,command = lambda: startTracking())
        setShape(runTracking_btn,RUNTRACKING_BTN_SHAPE,'nw')

    #   停止追蹤的按鈕
        runTracking_btn = tk.Button(window, text='停止追蹤' ,command = lambda: stopTracking())
        setShape(runTracking_btn,STOPTRACKING_BTN_SHAPE,'nw')
        
    #   停止追蹤的按鈕
        runTracking_btn = tk.Button(window, text='重新連接' ,command = lambda: reconnecting(chating_room))
        setShape(runTracking_btn,RECONN_BTN_SHAPE,'nw')

#     負責接收server端資料
#     會用threading呼叫執行持續監聽的動作
#     目前收到資料時只將資料顯示出來
    def recieve(self):
        global logger,close_recv
        while not close_recv:
            try:
                data = self.serversocket.recv(8192)
            except:
                break
            	
            try:
                jdata = json.loads(data)
            except:
                print("not json data")
                continue
            
            logger.info('got data=>' + jdata['cmd'])
            recv_cmd = jdata['cmd']
            print(recv_cmd)
            if recv_cmd == 'disconnect':
                break
            elif recv_cmd == 'say':            
                self.chat_text_list.append([0,jdata['name'],jdata['data']])
                update_chat_text()
        close_recv=False
        print("close recv")

#     程式剛開始運行時會執行run()
#     進行與SERVER連接的動作
    def run(self,ip,port,username):
        print(ip)
        print('test')
        print(port)
        self.serverip = ip
        self.serverport = port
        self.username = username
        
        try:
            print(1)
            conn_string="與伺服器(%s:%s)進行連接" % (self.serverip,self.serverport,)
            print(2)
            showText(self,"Rem",conn_string)
            print(3)
            self.serversocket.connect((self.serverip, int(self.serverport)))
            print(4)
            threading.Thread(target=self.recieve).start()
            print(5)
        except Exception as ex:
            print (ex)
            self.onconnectfailure()
            return False
        else:
            return True
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
    def sendTrackingData(self,string):
        msg = {'cmd':'track','name':self.username, 'data':string}
        jmsg = json.dumps(msg)
        print(jmsg)
        self.sendString(jmsg) 

#     getTime()負責從server取得當前時間
    def getTime(self):
        msg = {'cmd':'what time','name':'', 'data':''}
        jmsg = json.dumps(msg)
        self.serversocket.send(str.encode(jmsg))

#     disconnect()負責從server取得當前時間
    def disconnect(self):
        print("d1")
        #Closing connection with client
        self.serversocket.close()
        #Closing receive
        close_recv=True;
        print("d2")
        time.sleep(1)
        print("d3")
        
    def reconnect(self):
        while(True):
            print('reconn...')
            self.disconnect()
            if(self.run(self.serverip,self.serverport,self.username)):
                break
            else:
                print('reconn fail')
                time.sleep(1)
            pass
        print('reconn success')
        
            
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
    global chating_room,window,chat_text,logger
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

#    執行追蹤threading   
    th = threading.Thread(target=doCNNTracking, args = (cnn_args,))
    th.daemon=True
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