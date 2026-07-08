import json
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

# Paste your Google Cloud Console generated API key here (Starts with AIza)
API_KEY = "AQ.Ab8RN6IP6TTSG03bSb9I8HGp3lE0sQZE7PBwzSvSDWcNIwVwAA"

if not API_KEY:
    raise ValueError("API_KEY variable is empty. Please provide a valid key.")

# Initialize Gemini Client
client = genai.Client(api_key=API_KEY)


def analyze_plastic(image_bytes, mime_type):
    try:
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type
        )

        prompt = """
        You are an expert in waste management and plastic recycling.

        Analyze the uploaded image and identify the plastic type.

        Return ONLY valid JSON in this format:

        {
          "detected": true,
          "plastic_type": "PET",
          "recyclable": true,
          "details": "PET is commonly used in water bottles and can be recycled."
        }

        If no plastic is found:

        {
          "detected": false,
          "plastic_type": "",
          "recyclable": false,
          "details": ""
        }
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image_part],
            config={
                "response_mime_type": "application/json"
            }
        )

        result = json.loads(response.text)
        return result

    except Exception as e:
        print("ERROR:", str(e))
        return {
            "detected": False,
            "error": str(e)
        }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    image_bytes = file.read()
    mime_type = file.content_type

    result = analyze_plastic(image_bytes, mime_type)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)