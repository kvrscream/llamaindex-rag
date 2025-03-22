from flask import Flask, json, request
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def isUp():
  return 'flask is up'


@app.route('/upload', methods=['POST'])
def upload():
  try:
    file = request.files['file']
    file.save(os.path.join('docs', file.filename))
    response = {"message": "Arquivo enviado com sucesso!"}
    return json.dumps(response)
  except Exception as err:
    print(err)



if __name__ == "__main__":
  app.run(debug=True)