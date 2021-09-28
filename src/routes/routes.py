from src.utils import get_encoded_img
import src.config as config
from flask import render_template, request, redirect, url_for
from src import app
from src.predict import get_prediction
from src.utils import get_image, get_encoded_img


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if not image_file or not allowed_file(image_file.filename):
            return render_template("400.html")

        img_byte = image_file.read()
        img = get_image(img_byte)
        encoded_img = get_encoded_img(img_byte)
        img_data = f"data:image/jpeg;base64,{encoded_img.decode('utf-8')}"
        pred = get_prediction(img)
        return render_template("predict.html", prediction=pred, img_data=img_data)
    return redirect(url_for("index"))
