import os
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Spark_TTS')))


from Spark_TTS.SparkTTS import SparkTTS
import soundfile as sf # type: ignore


from flask import Flask, request, jsonify, send_file # type: ignore
from flask_cors import CORS # type: ignore
import os


import torch._dynamo
torch._dynamo.config.suppress_errors = True

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
logging.getLogger("torch._dynamo").setLevel(logging.CRITICAL)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


app = Flask(__name__)
CORS(app)


try:
    print("\nDownloading the model...")
    model_dir = "./Spark_TTS/pretrained_models/Spark-TTS-0.5B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparkTTS(model_dir, device)
    print("\nModel Downloaded and Integrated succesfully!!!\n")

except:
    print("Error in downloading model...")


@app.route("/")
def check_active():
    return jsonify({"status": "active"})


if not os.path.exists("results"):
    os.makedirs("results")


@app.route("/inf", methods=["POST"])
def TTS():
    print(request.form)
    if "audio" not in request.files or "text" not in request.form:
        return jsonify({"error": "Please provide both an audio file and text"}), 400

    audio_file = request.files["audio"]
    text = request.form["text"]

    try:
        print("Generating Inference...\n")

        with torch.no_grad():
            wav = model.inference(
                text=text,
                prompt_speech_path=audio_file,
                gender=None,
                pitch=None,
                speed=None,
            )

    except Exception as e:
        return jsonify({"status": f"Error in generating: {str(e)}"}), 500


    output_path = "./results/output.wav"
    sf.write(output_path, wav, samplerate=16000)

    return send_file(output_path, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4321, debug=False)