def train_model(classifier, xtrain, ytrain, xtest, ytest):
    # fit the training dataset on the classifier
    classifier.fit(xtrain, ytrain)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(xtest)
    
    return metrics.accuracy_score(ytest,predictions)