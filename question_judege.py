import pickle
query = ["Can you tell me something about Hong Kong","How is weather doday","What does ESG mean?"]
def judge(query):
    tfidf_path = 'tfidftransformer.pkl'
    tfidf = pickle.load(open(tfidf_path, "rb"))

    clf_lr_path = "clf_lr_model.pkl"
    cls = pickle.load(open(clf_lr_path, "rb"))
    return cls.predict(tfidf.transform(query))[0]
print(judge(query))