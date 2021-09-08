from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('Form.html')


@app.route('/', methods=['POST', 'GET'])
def predict():
    print(request.form)

    int_features = []
    pclass = request.form.get('pclass')
    gender = request.form.get('gender')
    age = request.form.get('age')
    sibsp = request.form.get('sibsp')
    parch = request.form.get('parch')
    fare = request.form.get('fare')

    int_features.append([pclass, age, fare, sibsp, parch, gender])

    print(int_features)
    prediction = model.predict_proba(int_features)
    print(prediction)
    output='{0:.{1}f}'.format(prediction[0][1]*100, 2)

    try:
        return render_template('Form.html', pred='Probability of surviving Titanic is {} %'.format(output))
    except ValueError:
        return 'Please enter valid values!'


if __name__ == '__main__':
    app.run(debug=True)
