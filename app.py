from flask import Flask,request,render_template
import pickle

#create flask app
app = Flask(__name__, template_folder='template')

model1 = pickle.load(open("model1.pkl", "rb"))
model2 = pickle.load(open("model2.pkl", "rb"))
model3 = pickle.load(open("model3.pkl", "rb"))
model4 = pickle.load(open("model4.pkl", "rb"))

@app.route("/")
def Home():
    return render_template(("out.html"))

@app.route("//predict", methods=["POST"])
def predict():
    datas=[[int(x) for x in request.form.values()]]
    #df = [np.array((datas))]
    pr1=(model1.predict(datas))
    pr2=(model2.predict(datas))
    pr3=(model3.predict(datas))
    pr4=(model4.predict(datas))


    return render_template("out.html",pr1 = "AGGREGATE EXPENDITURE IS {}crores".format(pr1),pr2 = "REVENUE EXPENDITURE IS {}crores".format(pr2) ,pr3="CAPITAL EXPENDITURE IS {}crores".format(pr3),pr4="SOCIAL EXPENDITURE IS {}crores".format(pr4))

if __name__=="__main__":
    app.run(debug=True)