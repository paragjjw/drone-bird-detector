from ultralytics import YOLO
import argparse
import cv2
from flask import Flask, render_template, request, flash
import os
import shutil
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")


@app.route("/")
def hello_world():
    return render_template('index.html')

# function for accessing rtsp stream
# @app.route("/rtsp_feed")
# def rtsp_feed():
    # cap = cv2.VideoCapture('rtsp://admin:hello123@192.168.29.126:554/cam/realmonitor?channel=1&subtype=0')
    # return render_template('index.html')


# Function to start webcam and detect objects

# @app.route("/webcam_feed")
# def webcam_feed():
    # #source = 0
    # cap = cv2.VideoCapture(0)
    # return render_template('index.html')

# function to get the frames from video (output video)

# def get_frame():
#     mp4_files = os.getcwd()+'/output.mp4'
#     video = cv2.VideoCapture(mp4_files)  # detected video path

#     while True:
#         success, image = video.read()
#         if not success:
#             break

#         ret, jpeg = cv2.imencode('.jpg', image)
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
#         # control the frame rate to display one frame every 100 milliseconds:
#         time.sleep(0.1)


# function to display the detected objects video on html page
# @app.route("/video_feed")
# def video_feed():
#     return Response(get_frame(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


# The display function is used to serve the image or video from the folder_path directory.
# @app.route('/<path:filename>')
# def display(filename):
#     folder_path = 'runs/detect/predict'
#     file_extension = filename.rsplit('.', 1)[1].lower()
#     environ = request.environ
#     if file_extension == 'jpg':
#         return send_from_directory(folder_path, filename, environ)
#     else:
#         return "Invalid file format"


@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files and request.files['file']:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)

            # predict_img.imgpath = f.filename
            # print("printing predict_img :::::: ", predict_img)

            file_extension = f.filename.rsplit('.', 1)[1].lower()

            model = YOLO('best.pt')  # Initializing the yolov8 pretrained model
            if file_extension == 'jpg':
                # Perform the detection
                if(os.path.exists(basepath+"/static/images/predict")):
                    shutil.rmtree(basepath+"/static/images/predict")
                detections = model.predict(
                    filepath, save=True, project="static/images", name="predict", imgsz=(640, 640))
                flash("File uploaded successfully")
                return render_template('index.html', image_path="images/predict/"+f.filename)

            elif file_extension == 'mp4':
                # if(os.path.exists("/static/videos/predict/output.mp4")):
                #     shutil.rmtree(basepath+"/static/videos/predict")
                video_path = filepath
                cap = cv2.VideoCapture(video_path)  # Capture video using cv2

                # get video dimensions
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                # Define the codec and create VideoWriter Object
                fourcc = 0x00000021
                out = cv2.VideoWriter(
                    'static/videos/predict/output.mp4', fourcc, fps, (frame_width, frame_height))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # retrieve YOLOv8 detection for the current video frame
                    results = model(frame)
                    print(results)
                    cv2.waitKey(1)

                    res_plotted = results[0].plot()
                    # cv2.imshow("result", frame)

                    # write the frame to the output video
                    out.write(res_plotted)

                    if(cv2.waitKey(1) == ord('q')):
                        break
                out.release()
                cap.release()
                cv2.destroyAllWindows()
                flash("File uploaded successfully")
                return render_template("index.html", video_path='videos/predict/output.mp4')
                # return send_from_directory("static/videos/predict", 'output.mp4', request.environ, conditional=True)
                # return render_template('index.html', video_path="videos/predict/output.mp4")
            else:
                flash("Invalid file format!")
                return render_template("index.html")
        else:
            flash("Please upload a file!")
            return render_template("index.html")
    else:
        return render_template("index.html")
