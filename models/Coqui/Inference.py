import os
import torch
from flask import Flask, request, jsonify, send_file
from TTS.api import TTS

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)

UPLOAD_FOLDER = "uploads"
OUTPUT_FILE = "output.wav"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/convert", methods=["POST"])
def convert_text_to_speech():
    if "audio" not in request.files or "text" not in request.form:
        return jsonify({"error": "Please provide both an audio file and text"}), 400

    audio_file = request.files["audio"]
    text = request.form["text"]

    audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(audio_path)

    tts.tts_to_file(text=text, speaker_wav=audio_path, language="en", file_path=OUTPUT_FILE)

    return send_file(OUTPUT_FILE, as_attachment=True)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=1234, debug=True)