

import flask
from flask import Flask, request, jsonify
# from models.handSegModel import HandSegModel
from jobs.BSAJob import BSAJob


app = Flask(__name__)

#  add job function here.
funcList = {
    "bsajob": BSAJob(),
}


@app.route('/')
def hello():
    return 'Hello World!'


@app.route('/api/help')
def showHelp():
    return f"loaded functions : {[k for k in funcList.keys()]}"


@app.route('/api/<func>', methods=['GET', 'POST'])
def reqfunc(func):
    if func not in funcList:
        return f"'{func}' not in func list."

    return jsonify(funcList[func].run(request))


def _model_init():
    pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6686)
