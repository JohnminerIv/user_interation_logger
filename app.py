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

import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("auth_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


application = app = Flask(__name__)
api = Api(app, version='1.0', title='MNIST Classification', description='CNN for Mnist')
ns = api.namespace('Make_School', description='Methods')

single_parser = api.parser()  # parsing args is one of the benefits of Flask-RESTPlus
single_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)

model = load_model('model.h5')


@ns.route('/prediction')
class CNNPrediction(Resource):
    """Uploads your data to the CNN"""
    @api.doc(parser=single_parser, description='Upload an mnist image')
    def post(self):
        args = single_parser.parse_args()
        args.file.save('posted_img.png')
        img = Image.open('posted_img.png')
        image_red = img.resize((28, 28))
        image = img_to_array(image_red)
        x = image.reshape((1, 784))
        x = x/255
        out = model.predict(x)
        r = str(np.argmax(out[0]))

        data = {
            'date': str(dt.datetime.now()),
            'file_name': args.file.filename,
            'result': r
        }
        db.collection('predictions').document(args.file.filename).set(data)

        return {'prediction': r, 'name': args.file.filename}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)