import joblib

def predict(data):
    knn=joblib.load('output_models\parkinsons_model.sav')
    return knn.predict(data)