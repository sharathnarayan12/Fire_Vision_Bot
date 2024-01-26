from flask import Flask, render_template, request
from ultralytics import YOLO
import os 

app = Flask(__name__ ,static_url_path='/static')
model = YOLO('fire_vision.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    result = model.predict(source=2, imgsz=640, conf=0.6, show=True)
    return result.imgs[0]
    

if __name__ == '__main__':
    # Create the main driver function
    port = int(os.environ.get("PORT", 5000)) # <-----
    app.run(host='0.0.0.0', port=port)       # <-----
    # # app.run(debug=True)
    # app.run(host='0.0.0.0', port='5000')