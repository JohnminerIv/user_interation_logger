import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from werkzeug.datastructures import FileStorage

import datetime as dt 

from flask_restplus import Api, Resource, fields
from flask import Flask, request, jsonify

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder

import firebase_admin
from firebase_admin import credentials, firestore
import sys

cred = credentials.Certificate("auth_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


application = app = Flask(__name__)
api = Api(app, version='1.0', title='Solar panel anomaly', description='CNN for solar panel images')
ns = api.namespace('Raptor maps project', description='Methods')

single_parser = api.parser()  # parsing args is one of the benefits of Flask-RESTPlus
single_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)

model = load_model('model.h5')
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy')


@ns.route('/prediction')
class CNNPrediction(Resource):
    """Uploads your data to the CNN"""
    @api.doc(parser=single_parser, description='Upload an mnist image')
    def post(self):
        args = single_parser.parse_args()
        args.file.save('posted_img.png')
        img = Image.open('posted_img.png')
        print(sys.getsizeof(img))

        image_red = img.resize((40, 24))
        image = img_to_array(image_red)
        x = image.reshape((1, 40, 24, 1))
        x = x/255
        out = model.predict(x)
        r = np.argmax(out[0])
        r = encoder.inverse_transform([r])[0]
        print(sys.getsizeof(image))
        ls = list(image.reshape(-1))
        ls = ' '.join(map(str, ls))
        print(sys.getsizeof(ls))

        data = {
            'date': str(dt.datetime.now()),
            'file_name': args.file.filename,
            'result': r,
            'image': ls
        }
        db.collection('predictions').document(args.file.filename).set(data)

        return {'prediction': r, 'name': args.file.filename}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)