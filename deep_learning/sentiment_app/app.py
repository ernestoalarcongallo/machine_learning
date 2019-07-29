from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
from model import Model

app = Flask(__name__)

class SentimentForm(Form):
    sayhello = TextAreaField('',[validators.DataRequired()])

@app.route('/')
def index():
    form = SentimentForm(request.form)
    return render_template('main.html', form=form)

@app.route('/hello', methods=['POST'])
def hello():
    model = Model()
    form = SentimentForm(request.form)
    if request.method == 'POST' and form.validate():
        name = request.form['sayhello']
        predictions = model.predict(name)
        if predictions is None:
            sentiment='Sorry, I can\'t predict this, I am still learning... 🙏'
        else:
            sentiment=predictions[0]
        return render_template('sentiment.html', name=sentiment)
    return render_template('main.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)