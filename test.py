from flask import Flask, render_template
app = Flask(__name__)

@app.route('/test')
def test():
    return '<h1 style="color:white;background:black">Flask is Working!</h1>'

@app.route('/testhome')
def testhome():
    return render_template('home.html')

@app.route('/testform')
def testform():
    return render_template('index.html')

app.run(debug=True)
