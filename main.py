from flask import Flask, render_template, request
import os

UPLOAD_FOLDER = 'static/images'
START_IMG = 'static/images/jianyang.jpeg'

app = Flask(__name__, template_folder='template')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        if 'img-file' not in request.files:
            return 'there is no img-file in form!'
        img_file = request.files['img-file']
        img_file.filename = "predict_img.png"
        path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
        img_file.save(path)
        predict = food.predictData(img_file.filename)
        return render_template('index.html', image=path, prediction=predict)

    return render_template('index.html', image=START_IMG)

if __name__ == '__main__':

    try:
        from model import SeeFood

        """if you want to import and train new data to dataset"""
        # all images/data has to be inside argument folder located in /data/train/{argument}
        # (when creating dataset, function will create dir and csv file with same name inside dir)
        #food = SeeFood().createData('hotdog')
        
        # if dataset with multiple folders, i.g hotdog, pizza, hamburger
        #food = SeeFood().createData(['hotdog', 'pizza', 'hamburger'])

        """if dataset & dir already exists"""
        # directory with csv file as init argument
        food = SeeFood('hotdog_pizza_hamburger')

        app.debug = True 
        app.run(host='0.0.0.0', port='1312')

    except (ImportWarning, ImportError, FileNotFoundError) as e:
        print(e)