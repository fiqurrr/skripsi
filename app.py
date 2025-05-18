from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

dic = {
    0: 'ABSTRAK HALUAN JALUR + PENDAYUNG + TAKULUAK BAREMBAI',
    1: 'ABSTRAK TAKULUAK BAREMBAI+PENDAYUNG',
    2: 'BAMBU+TAKULUAK BAREMBAI TABUR',
    3: 'BUJANG DARA KUANSING',
    4: 'BUNGA JALUR + DAUN SEDINGIN',
    5: 'BUNGA PAKIS+PENDAYUNG',
    6: 'CAGAK JALUR BAPACU',
    7: 'CALEMPONG',
    8: 'CALEMPONG+DAYUNG',
    9: 'CALEMPONG+TAKULUAK BAREMBAI',
    10: 'CARANO',
    11: 'DAYUNG',
    12: 'HALUAN JALUR + PENDAYUNG',
    13: 'HALUAN JALUR + TAKULUAK BAREMBAI',
    14: 'HALUAN JALUR + TUGU JALUR',
    15: 'HARIMAU',
    16: 'JALUR',
    17: 'JALUR + PONDIANG',
    18: 'JALUR + TAKULUAK BAREMBAI',
    19: 'JAMBAU',
    20: 'KONJI BARAYAK',
    21: 'LANCANG KUNING',
    22: 'PENDAYUNG SILANG + TAKULUAK BAREMBAI',
    23: 'PERAHU BAGANDUANG',
    24: 'PERAHU BAGANDUANG + BUNGA',
    25: 'PERAHU BAGANDUANG + DAYUNG',
    26: 'PERAHU BAGANDUANG + PAYUNG',
    27: 'PERAHU BAGANDUANG + TAKULUAK BAREMBAI',
    28: 'PONDIANG',
    29: 'PULUIK KUCUANG',
    30: 'SILEK',
    31: 'TAKULUAK BAREMBAI',
    32: 'TUAI PADI',
    33: 'TUGU JALUR',
    34: 'URANG MENYENTAK',
}

model = load_model('model_cnn_batik.h5')

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224, 224))
    i = image.img_to_array(i) / 255.0
    i = i.reshape(1, 224, 224, 3)
    pred = model.predict(i)
    p = np.argmax(pred, axis=1)
    return dic[int(p[0])]

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("classification.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = os.path.join("static", img.filename)
        img.save(img_path)

        p = predict_label(img_path)

        return render_template("classification.html", prediction=p, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
