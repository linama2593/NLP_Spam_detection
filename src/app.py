from flask import Flask, request, render_template #pip install flask gunicorn
from pickle import load

app = Flask(__name__)
model=load(open('/workspaces/NLP_Spam_detection/models/svc_c80_ovo_degr14_seed42.pk','rb'))

class_dict={
    '1': 'URL is Spam',
    '0': 'URL is not Spam'
}

@app.route("/", methods=['GET', 'POST'])

def index():
    if request.method=='POST':
        url=str(request.form['val1'])
        data=[url]
        prediction=str(model.predict(data)[0])
        pred_class=class_dict[prediction]
    else:
        pred_class=None
    
    return render_template('index.html', prediction=pred_class)

