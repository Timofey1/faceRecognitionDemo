import pymongo
from functions import *
from annoy import AnnoyIndex
annoyInd = AnnoyIndex(512, "euclidean")
annoyInd.load("annoyIndFile.ann")

print(annoyInd.get_n_items())

db_cli = pymongo.MongoClient("localhost", 27017)
FacesTrain = db_cli["facesInfo"]
train_collection = FacesTrain["facesInfo"]

FacesTest = db_cli["facesInfoTest"] # DB with test faces
test_collection = FacesTest["facesInfo"]
print("=" * 50)
for metric in ['euclidean', 'cosine', 'euclidean_l2']:
    tp, tn, fp, fn = 0, 0, 0, 0
    print(metric)
    for test_face in test_collection.find():
        test_face_vector = test_face["vector"]
        test_face_label = test_face["name"]
        predict = annoyInd.get_nns_by_vector(test_face_vector, 1)[0]

        train_face = train_collection.find_one({"faceId": predict+1})
        train_face_vector = train_face["vector"]
        train_face_label = train_face["name"]

        distance, threshold = compareEmbeddings(test_face_vector, train_face_vector, metric)

        if test_face_label == train_face_label:
            if distance < threshold:
                tp += 1
            else:
                tn += 1
        else:
            if distance > threshold:
                fp += 1
            else:
                fn += 1

    TPR = tp/(tp+fn)
    FPR = fp/(tn+fp)

    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    print("tp, tn, fp, fn")
    print(tp, tn, fp, fn)
    print("_"*20)
    print("TPR:", TPR, "FPR:", FPR)
    print("_"*20)
    print("precision:", precision, "recall:", recall)
    print("="*50)
