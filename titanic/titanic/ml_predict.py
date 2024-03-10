def prediction_model(pclass, sex, age, sibsp, parch, fare, embarked, title):
    import pickle
    x = [[pclass, sex, age, sibsp, parch, fare, embarked, title]]
    with open("titanic_model.pkl", "rb") as f:
        randomforest = pickle.load(open('titanic_model.pkl', 'rb'))
        prediction = randomforest.predict(x)
        if prediction == 0:
            prediction = 'Not Survived'
        elif prediction == 1:
            prediction = 'Survived'
        else:
            prediction = 'Error'
        return prediction
