import json
import os
from flask import Flask, render_template, request, jsonify

# 1. READ THE API KEY FROM THE TEXT FILE
try:
    with open("api_key.txt", "r") as file:
        # .strip() removes any accidental hidden spaces or newlines from the file
        os.environ["GEMINI_API_KEY"] = file.read().strip()
except FileNotFoundError:
    raise ValueError("ERROR: The file 'api_key.txt' was not found in your project directory!")

# Clear any conflicting global legacy GCP auth variables if they exist
os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)

from google import genai
from google.genai import types

app = Flask(__name__)

# 2. INITIALIZE CLIENT
# Leaving Client() blank allows it to natively fetch the variable we just loaded into os.environ
client = genai.Client()


def analyze_plastic(image_bytes, mime_type):
    try:
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type
        )

        prompt = """
        You are an expert computer vision model specializing in waste management and plastic recycling.

        Analyze the uploaded image carefully. Identify if there is a plastic item, determine its specific plastic type/resin identification code, and assess its general recyclability.

        Look for characteristics matching these categories:
        1. PET (Polyethylene Terephthalate) - Water/soda bottles, transparent jars.
        2. HDPE (High-Density Polyethylene) - Milk jugs, shampoo bottles, detergent containers.
        3. PVC (Polyvinyl Chloride) - Pipes, credit cards, certain clear food wraps.
        4. LDPE (Low-Density Polyethylene) - Squeezable bottles, grocery bags, flexible wraps.
        5. PP (Polypropylene) - Yogurt cups, syrup bottles, bottle caps, straws.
        6. PS (Polystyrene) - Styrofoam, disposable cups, plastic cutlery, egg cartons.
        7. OTHER - Acrylic, fiberglass, nylon, multi-layer plastics, or composite resin.

        You MUST return ONLY a valid JSON object. Do not include markdown wraps like ```json. 
        Dynamically fill out the fields based on the actual object identified in the image:

        If a plastic object is detected, populate it like this template:
        {
          "detected": true,
          "plastic_type": "INSERT_ACRONYM_HERE (e.g., PET, HDPE, PP, LDPE, etc.)",
          "recyclable": true, 
          "details": "Provide a specific 2-3 sentence breakdown of what the item is, what type of plastic it is made of, and real-world tips on how to properly dispose of or recycle it."
        }

        If absolutely no plastic item is found in the image:
        {
          "detected": false,
          "plastic_type": "",
          "recyclable": false,
          "details": ""
        }
        """

        # Call Gemini 2.5 Flash
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
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