from flask import Flask, render_template, Response, request
import cv2
import skimage as ski
import numpy as np
import face_recognition as fr
import numpy as np
import pandas as pd
import time

app = Flask(__name__)

fname='feature.csv'
counter=0
names=[]
feats=[]

#for registration

def generate_frames(name):
    #name=request.form.get('user_input')
    global names
    global feats
    global counter
    try:
        df=pd.read_csv(fname)
    except:
        df=pd.DataFrame({'name':[],'enc':[]})

    vid = cv2.VideoCapture(0)  # Open the camera (0 represents default camera)
    while True:
        ack ,img = vid.read()
        if not ack:
            break
        else:           
            fd=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            vid=cv2.VideoCapture(0)
            counter=0
            while True:
                ack,img=vid.read()
                if ack:
                    faces=fd.detectMultiScale(img,1.2,2,minSize=(150,150))
                    if len(faces)==1:
                        x,y,w,h=faces[0]
                        face_img=img[y:y+h,x:x+w,:].copy()
                        face_enc=fr.face_encodings(face_img)

                        if len(face_enc)==1:
                            counter+=1

                            names+=[name]
                            feats +=[face_enc[0].tolist()]

                        if counter==20:
                            f=pd.DataFrame({'name':names,'enc':feats})
                            df=pd.concat([df,f],axis=0,ignore_index=True)
                            df.to_csv(fname)
                            break     

                    ack, buffer = cv2.imencode('.jpg', img)
                    img = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

# for recogination

def generate_frames1():
    fname='feature.csv'
    try:
        df=pd.read_csv(fname)
    except:
        print('Data base not found')
    else:
        fd=cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')
        while True:
            vid=cv2.VideoCapture(0)
            ack,img=vid.read()
            if ack:
            #do the entire processing
                faces=fd.detectMultiScale(img,1.2,2,minSize=(150,150))
            #faces= [(x,y,w,h),(x2,y2,w2,h2)]
                face_enc = []
                matches = []

                if len(faces)==1:
                    x,y,w,h=faces[0]
                    faces_img=img[y:y+h,x:x+w,:].copy()
                    face_enc=fr.face_encodings(faces_img)  

             #for recogination part main code is here
                if len(face_enc)==1:
                    feats_data=df['enc'].apply(lambda x:eval(x)).values.tolist()
                    matches=fr.compare_faces(face_enc,np.array(feats_data))

                if True in matches:
                    match_ind=matches.index(True)
                    name_find=df['name'][match_ind]
                else:
                    name_find='Unknow'
                cv2.putText(img,str(name_find),(150,150)  ,   
                        cv2.FONT_HERSHEY_PLAIN,10,(0,0,255),5)
                
            ack, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

    vid.release()

#by default home page

@app.route('/', methods=['GET', 'POST'])
def index():     
    return render_template('index.html')

#by default home page camera on

@app.route('/video_feed')
def video_feed():
    name = request.form.get('user_input')
    return Response(generate_frames(name), mimetype='multipart/x-mixed-replace; boundary=frame')

#by face recogination page 

@app.route('/face_rec')
def new_page():
    return render_template('face_rec.html')

#by default face recogination page open camera

@app.route('/video_done1')
def video_done1():    
    return Response(generate_frames1(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
