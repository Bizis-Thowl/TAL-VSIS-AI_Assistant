from pickle import load

def retrieve_model(filename):
    with open(f"learning/models/{filename}", "rb") as f:
        clf = load(f)
        
    return clf
