from flask import Flask, request
app = Flask(__name__)

import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    # Check if the request contains a file
    if 'file' not in request.files:
        return "No file found in the request"

    audio_file = request.files['file']
    audio_data = audio_file.read()

    # Make a request to the Whisper API
    response = openai.Transcriber.transcribe(audio_data)

    # Get the transcribed text from the response
    transcription = response['text']

    # Return the transcribed text as the API response
    return transcription

if __name__ == '__main__':
    # print("OPENAI_API_KEY : ", end = ' ' )
    # print(os.environ["OPENAI_API_KEY"])
    app.run()
