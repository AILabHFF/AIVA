import os
import cv2
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from visualprocessing.frame_processor import VisualProcessor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

def generate_frames(video_file, scale, method, threshold, ksize, object_minsize, object_maxsize):
    cap = cv2.VideoCapture(video_file)
    frameProcessor = VisualProcessor(scale=scale, method=method, threshold=threshold, ksize=ksize, object_minsize=object_minsize, object_maxsize=object_maxsize, filename=video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame_num = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            frame = frameProcessor.process_frame(frame, current_frame_num)
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            progress = int((current_frame_num / total_frames) * 100)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                   b'Content-Type: text/event-stream\r\n\r\n' + b'event: progress\n' + f'data: {progress}\n\n'.encode())
            
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    video_file = request.args.get('video_file')
    scale = int(request.args.get('scale'))
    method = request.args.get('method')
    threshold = int(request.args.get('threshold'))
    ksize = int(request.args.get('ksize'))
    object_minsize = int(request.args.get('object_minsize'))
    object_maxsize = int(request.args.get('object_maxsize'))
    return Response(generate_frames(os.path.join(app.config['UPLOAD_FOLDER'], video_file), scale, method, threshold, ksize, object_minsize, object_maxsize), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['video_file']
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        return jsonify({'video_file': filename, 'video_file': filename, 'scale': request.form['scale'], 'method': request.form['method'], 'threshold': request.form['threshold'],
                        'ksize': request.form['ksize'], 'object_minsize': request.form['object_minsize'], 'object_maxsize': request.form['object_maxsize']})
    else:
        return redirect(url_for('index'))
    

if __name__ == '__main__':
    #app.run(debug=False)
    from waitress import serve
    import webbrowser

    webbrowser.open('http://127.0.0.1:5000', new=1)
    print('Running on http://127.0.0.1:5000')
    serve(app, host="0.0.0.0", port=5000)
