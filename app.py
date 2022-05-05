from flask import Flask, render_template, request, url_for
import os, shutil
from predictions import pred_price
import gdown

app = Flask(__name__) 

@app.route("/")
def hello():   

    folder = 'static'
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        os.makedirs(folder)

    return render_template("index.html")

@app.route("/loading", methods = ['GET','POST'])
def Upload():
    if request.method == 'POST':
        image = request.files["fileToUpload"]
        path = os.path.join('static', image.filename)
        print(path)
        image.save(path)        

        return render_template("loading.html", img = image.filename)

@app.route("/processing",  methods = ['GET','POST'])
def result():
    #img_filename = str(request.args.to_dict(flat=False)['data1'][0])
    
    img_filename = str(request.args.to_dict(flat=False)['data'][0])
    path = os.path.join('static', img_filename)
    #print(img_filename)
    url1 = 'https://drive.google.com/uc?export=download&id=1-yqDSi6kZ-aOuOJQdhvVdis6W-sIMT10'
    output1 = 'models/model_final_1.pth'
    url2 = 'https://drive.google.com/uc?export=download&id=1-HEhFAJzFSveKSLhxdV80IZG96S9bNHT'
    output2 = 'models/model_final_2.pth'
    if not os.path.exists('models'):
        os.makedirs('models')
    gdown.download(url1, output1, quiet=False)
    gdown.download(url2, output2, quiet=False)
    print("Models Downloaded")
    print("Making Predictions")
    # This is where your time-consuming stuff can take place (sql queries, checking other apis, etc)
    price, pred_path = pred_price(path) # To simulate something time-consuming, I've tested up to 100 seconds
    # You can return a success/fail page here or whatever logic you want
    return render_template('result.html',img = img_filename, pre = pred_path ,pri = price)


if __name__ == "__main__":
    app.run()