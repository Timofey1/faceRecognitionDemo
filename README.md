# Face Recognition Demo
 - facenet(trained on vggface2 dataset)
 - MTCNN face detector
 - mongoDB for storing facial embeddings
 - spotify annoy for indexing faces and finding simular faces 
 - Flask framework + Flask-RESTful for REST API

### Installation and startup
    pip install -r requirements.txt
    python app.py / flask run 

### Options
    - Detector only
    - Compare two loaded faces
    - Add/Find face in DataBase (install mongoDB first)
