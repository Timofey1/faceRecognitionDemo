import io
import os
import re
from flask import Flask, render_template, request, send_from_directory
from flask_restful import Api, Resource, reqparse
import werkzeug
import cv2
from PIL import Image
import ast
import numpy as np
import annoy
from flask_pymongo import PyMongo
from facenet_pytorch import MTCNN, InceptionResnetV1
from functions import *
from torchvision.transforms import functional as F
from facenet_pytorch.models.mtcnn import fixed_image_standardization

app = Flask(__name__)
api = Api(app)
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

model_inp = 160
MODEL = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=model_inp)

app.config["MONGO_URI"] = "mongodb://localhost:27017/facesInfo"
mongo = PyMongo(app)

UPLOAD_API_DIRECTORY = 'api_uploads'


class FaceRec(Resource):
    def post(self):
        image_post_args = reqparse.RequestParser()
        image_post_args.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        image_post_args.add_argument("name", type=str, help="Identity name is required", required=True)
        data = image_post_args.parse_args()

        if data['file'] == "":
            return {'message': 'No file found', 'status': 'error'}, 404

        photo = data['file'].read()
        print(type(photo))
        if photo:
            img = Image.open(io.BytesIO(photo))
            try:
                img_cropped = mtcnn(img)
                embedding = MODEL(img_cropped.unsqueeze(0))[0].tolist()
                print("embedding length = ", len(embedding))
            except Exception:
                return {'message': 'faces not found', 'status': 'error'}, 404

            faceName = re.sub('[^a-zA-Zа-яА-Я ]+', '', data["name"]).title()
            faceId = mongo.db.facesInfo.count()+1
            inserted_res = mongo.db.facesInfo.insert(
                {"faceId": faceId, "name": faceName, "vector": embedding})
            return {'message': 'photo uploaded', 'status': 'success', 'id': faceId}, 201
        return {'message': 'Something went wrong', 'status': 'error'}, 404

    def get(self):
        image_get_args = reqparse.RequestParser()
        image_get_args.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        data = image_get_args.parse_args()

        if data['file'] == "":
            return {'message': 'No file found', 'status': 'error'}, 404

        photo = data['file'].read()

        if photo:
            img = Image.open(io.BytesIO(photo))
            try:
                img_cropped = mtcnn(img)
                vector = MODEL(img_cropped.unsqueeze(0))[0].tolist()
                print("embedding length = ", len(vector))
            except Exception:
                return {'message': 'faces not found', 'status': 'error'}, 404

            annoyInd = annoy.AnnoyIndex(512, "euclidean")
            annoyInd.load("annoyIndFile.ann")
            res = annoyInd.get_nns_by_vector(vector, 3)
            names = {}
            for i in res:
                name = mongo.db.facesInfo.find_one({"faceId": i + 1})["name"]
                saved_vector = mongo.db.facesInfo.find_one({"faceId": i + 1})["vector"]
                verification_result = {}
                for metric in ['euclidean', 'cosine', 'euclidean_l2']:
                    distance, threshold = compareEmbeddings(vector, saved_vector, metric)
                    if distance < threshold:
                        identified = "True"
                    else:
                        identified = "False"
                    resp_obj = {
                        "verified": identified
                        , "distance": distance
                        , "max_threshold_to_verify": threshold
                    }
                    verification_result[metric] = resp_obj
                names[name] = verification_result
            return names, 201
        return {'message': 'Something went wrong', 'status': 'error'}, 404



class DBmethods(Resource):
    def get(self, faceId):
        res = mongo.db.facesInfo.find_one({"faceId": faceId})
        if res is None:
            return {'message': 'No such id in database', 'status': 'error'}, 404
        res["faceId"] = str(res["faceId"])
        return res, 200

    def delete(self, faceId):
        res = mongo.db.facesInfo.delete_one({"faceId": faceId})
        mongo.db.facesInfo.find_and_modify(query={"faceId": {"$gt": faceId}},
                                           update={"$inc": {"faceId": -1}})
        return res.raw_result, 204


class MtcnnApi(Resource):
    def post(self):
        retina_post_args = reqparse.RequestParser()
        retina_post_args.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        data = retina_post_args.parse_args()

        if data['file'] == "":
            return {'message': 'No file found', 'status': 'error'}, 404

        photo = data['file'].read()

        if photo:
            img = Image.open(io.BytesIO(photo))
            area, score, landm = mtcnn.detect(img, landmarks=True)
            obj = []
            for i in range(len(area)):
                tmp = {}
                tmp["box"] = area[i].astype("int64").tolist()
                tmp['confidence'] = str(score[i])
                tmp["keypoints"] = {'left_eye': landm[i][0].astype("int64").tolist(),
                                    'right_eye': landm[i][1].astype("int64").tolist(),
                                    'nose': landm[i][2].astype("int64").tolist(),
                                    'mouth_left': landm[i][3].astype("int64").tolist(),
                                    'mouth_right': landm[i][4].astype("int64").tolist()}
                obj.append(tmp)
            res = {}
            for i, face in enumerate(obj):
                res["face_"+str(i+1)] = face
            if len(res) == 0:
                return {"message": "faces not found"}
            else:
                print(res)
                return res
        return {'message': 'Something went wrong', 'status': 'error'}, 404


api.add_resource(FaceRec, '/photoUpl')
api.add_resource(DBmethods, '/dbApi/<int:faceId>')
api.add_resource(MtcnnApi, '/mtcnnFaceApi')


@app.route("/")
def index():
    return '''
    <p><a href="retina">Детектор</a></p>
    <p><a href="verify">Сравнение</a></p>
    <p><a href="faces">Добавление/Поиск</a></p>
    '''


@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory('uploads', filename)


@app.route("/retina")
def retina():
    return render_template('retina1.html')


@app.route("/retina", methods=['POST'])
def retina_post():
    image = request.files["file"]
    if image.filename != '':
        image.save(os.path.join('uploads', "retina.jpg"))
    else:
        Pimg = Image.open(os.path.join('uploads', "ind1.jpg"))
        Pimg.save("uploads/retina.jpg")

    img_pth = os.path.join('uploads', "retina.jpg")
    img = Image.open(img_pth)
    area, score, landm = mtcnn.detect(img, landmarks=True)
    obj = []
    for i in range(len(area)):
        tmp = {}
        tmp["box"] = area[i].astype("int64").tolist()
        tmp['confidence'] = score[i]
        tmp["keypoints"] = {'left_eye': landm[i][0].astype("int64").tolist(),
                            'right_eye': landm[i][1].astype("int64").tolist(),
                            'nose': landm[i][2].astype("int64").tolist(),
                            'mouth_left': landm[i][3].astype("int64").tolist(),
                            'mouth_right': landm[i][4].astype("int64").tolist()}
        obj.append(tmp)

    retinaLikeData = {}
    for i, face in enumerate(obj):
        tmp = {}
        tmp["facial_area"] = face["box"]
        tmp["score"] = face["confidence"]
        tmp["landmarks"] = face["keypoints"]
        retinaLikeData["face_" + str(i + 1)] = tmp
    try:
        img = cv2.imread(img_pth)
        for face in retinaLikeData.keys():
            facial_area = retinaLikeData[face]["facial_area"]
            cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255, 255, 255), 1)
        cv2.imwrite("uploads/bbox.jpg", img)
        retinaLikeData = str(retinaLikeData)
        retinaLikeData = retinaLikeData.replace("(", "[")
        retinaLikeData = retinaLikeData.replace(")", "]")
        return render_template("retina2.html", objects=str(retinaLikeData), fc=len(obj))
    except Exception:
        cv2.imwrite("uploads/bbox.jpg", img)
        print("No faces detected")
        return render_template("retina2.html", objects={'error': 'No faces detected'}, fc=0)


@app.route('/verify')
def verify():
    return render_template('verify1.html')


@app.route('/verify', methods=['POST'])
def verify_post():
    image1 = request.files["img1"]
    image2 = request.files["img2"]
    if image1.filename != '':
        image1.save(os.path.join('uploads', "image1.jpg"))
    else:
        Pimg = Image.open(os.path.join('uploads', "ind1.jpg"))
        Pimg.save("uploads/image1.jpg")

    if image2.filename != '':
        image2.save(os.path.join('uploads', "image2.jpg"))
    else:
        Pimg = Image.open(os.path.join('uploads', "ind2.jpg"))
        Pimg.save("uploads/image2.jpg")
    img1 = Image.open(os.path.join('uploads', "image1.jpg"))
    img2 = Image.open(os.path.join('uploads', "image2.jpg"))

    img_cropped1 = mtcnn(img1)
    img_cropped2 = mtcnn(img2)

    img_embedding1 = MODEL(img_cropped1.unsqueeze(0))[0].tolist()
    img_embedding2 = MODEL(img_cropped2.unsqueeze(0))[0].tolist()

    verification_result = {}
    for metric in ['euclidean', 'cosine', 'euclidean_l2']:
        distance, threshold = compareEmbeddings(img_embedding1, img_embedding2, metric)
        if distance < threshold:
            identified = "True"
        else:
            identified = "False"
        resp_obj = {
            "verified": identified
            , "distance": distance
            , "max_threshold_to_verify": threshold
        }
        verification_result[metric] = resp_obj
    return render_template("verify2.html", res=str(verification_result))

@app.route('/faces')
def check():
    return render_template("retina1.html")


@app.route("/faces", methods=['POST'])
def check_post():
    image = request.files["file"]
    if image.filename != '':
        image.save(os.path.join('uploads', "chooseFaceFrom.jpg"))
    else:
        Pimg = Image.open(os.path.join('uploads', "ind1.jpg"))
        Pimg.save("uploads/chooseFaceFrom.jpg")

    img_pth = os.path.join('uploads', "chooseFaceFrom.jpg")

    img = Image.open(img_pth)
    width, _ = img.size
    area, score, landm = mtcnn.detect(img, landmarks=True)
    obj = []
    for i in range(len(area)):
        tmp = {}
        tmp["box"] = area[i].astype("int64").tolist()
        tmp['confidence'] = score[i]
        tmp["keypoints"] = {'left_eye': landm[i][0].astype("int64").tolist(),
                            'right_eye': landm[i][1].astype("int64").tolist(),
                            'nose': landm[i][2].astype("int64").tolist(),
                            'mouth_left': landm[i][3].astype("int64").tolist(),
                            'mouth_right': landm[i][4].astype("int64").tolist()}
        obj.append(tmp)

    retinaLikeData = {}
    for i, face in enumerate(obj):
        tmp = {}
        tmp["facial_area"] = face["box"]
        tmp["score"] = face["confidence"]
        tmp["landmarks"] = face["keypoints"]
        retinaLikeData["face_"+str(i+1)] = tmp
    return render_template("check2.html", objects=retinaLikeData, width=width)


@app.route("/crop", methods=['POST'])
def crop_post():
    obj = ast.literal_eval(request.form.get("obj"))
    face = request.form.get("faceNum")
    region = obj[face]["facial_area"]
    img = cv2.imread(os.path.join('uploads', "chooseFaceFrom.jpg"))
    selected_face = img[region[1]:region[3], region[0]:region[2]]
    cv2.imwrite("uploads/selected_face.jpg", selected_face)
    aligned_face = alignment_procedure(selected_face, obj[face]["landmarks"]["left_eye"],
                                                    obj[face]["landmarks"]["right_eye"])
    aligned_face = cv2.resize(aligned_face, (model_inp, model_inp))
    cv2.imwrite("uploads/aligned_face.jpg", aligned_face)
    return render_template("crop.html")


@app.route("/doneCrop", methods=['POST'])
def done_crop_post():
    face = re.sub('[^a-zA-Zа-яА-Я ]+', '', request.form.get("person")).title()
    print(">>> face:", face)
    img = Image.open(os.path.join('uploads', "selected_face.jpg"))
    out_img = crop_resize(img, 160)
    im_tensor = F.to_tensor(np.float32(out_img))
    im_tensor = fixed_image_standardization(im_tensor)
    embedding = MODEL(im_tensor.unsqueeze(0))[0].tolist()

    # mongoDB
    faceId = mongo.db.facesInfo.count()+1
    inserted_res = mongo.db.facesInfo.insert(
        {"faceId": faceId, "name": face, "vector": embedding})
    print(">>> faceId:", faceId)

    annoyInd = annoy.AnnoyIndex(512, "euclidean")
    for i, row in enumerate(mongo.db.facesInfo.find()):
        vector = row["vector"]
        annoyInd.add_item(i, vector)
    annoyInd.build(15)
    annoyInd.save("annoyIndFile.ann")
    return render_template("doneUpload.html", faceId=faceId)


@app.route("/find", methods=['POST'])
def find_post():
    kx = int(request.form.get("number"))
    img = Image.open(os.path.join('uploads', "selected_face.jpg"))
    out_img = crop_resize(img, 160)
    im_tensor = F.to_tensor(np.float32(out_img))
    im_tensor = fixed_image_standardization(im_tensor)
    vector = MODEL(im_tensor.unsqueeze(0))[0].tolist()
    annoyInd = annoy.AnnoyIndex(512, "euclidean")
    annoyInd.load("annoyIndFile.ann")
    res = annoyInd.get_nns_by_vector(vector, kx)
    names = {}
    for i in res:
        name = mongo.db.facesInfo.find_one({"faceId": i+1})["name"]
        saved_vector = mongo.db.facesInfo.find_one({"faceId": i+1})["vector"]
        verification_result = {}
        for metric in ['euclidean', 'cosine', 'euclidean_l2']:
            distance, threshold = compareEmbeddings(vector, saved_vector, metric)
            if distance < threshold:
                identified = "True"
            else:
                identified = "False"
            resp_obj = {
                "verified": identified
                , "distance": distance
                , "max_threshold_to_verify": threshold
            }
            verification_result[metric] = resp_obj
        names[name] = verification_result
    return render_template("find.html", names=names)


if __name__ == '__main__':
    app.run()
