import joblib 

def predict(data):
    clf = joblib.load('output_model/kn_model.sav')
    return clf.predict(data)