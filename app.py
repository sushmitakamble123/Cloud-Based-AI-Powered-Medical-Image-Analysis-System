from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import boto3
import uuid   

app = Flask(__name__)


model = tf.keras.models.load_model("medical_model.h5")

CLASS_NAMES = [
    "BONE_FRACTURE",
    "BRAIN_NORMAL",
    "BRAIN_TUMOR",
    "BONE_NORMAL",
    "CHEST_NORMAL",
    "CHEST_PNEUMONIA",
    "SPINE_ISSUE",
    "SPINE_NORMAL"
]


UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


S3_BUCKET = "medical-image-project-bucket"
S3_REGION = "ap-south-1"

s3 = boto3.client("s3", region_name=S3_REGION)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]

        if file.filename == "":
            return "No file selected"

        
        unique_filename = f"{uuid.uuid4()}_{file.filename}"

        filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filepath)

        
        img = image.load_img(filepath, target_size=(224,224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        
        pred = model.predict(img_array)
        label = CLASS_NAMES[np.argmax(pred)]

        body_part, condition = label.split("_")

    
        try:
            s3_key = f"{body_part}/{condition}/{unique_filename}"
            s3.upload_file(filepath, S3_BUCKET, s3_key)

            s3_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
        except Exception as e:
            print("S3 upload failed:", e)
            s3_url = None

        
        s3_console_url = (
            f"https://s3.console.aws.amazon.com/s3/buckets/{S3_BUCKET}"
            f"?region={S3_REGION}&prefix={body_part}/{condition}/&showversions=false"
        )

        return render_template(
            "result.html",
            body_part=body_part,
            condition=condition,
            image=filepath,
            s3_url=s3_url,
            s3_console_url=s3_console_url
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
