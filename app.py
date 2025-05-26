# app.py
import os, cv2, json
import numpy as np
import gradio as gr
from deepface import DeepFace
from scipy.spatial.distance import cosine

DB_PATH = "face_db.npz"


# ---------- tiny face-bank helpers ----------
def load_db():
    if os.path.exists(DB_PATH):
        data = np.load(DB_PATH, allow_pickle=True)
        return dict(zip(data["names"], data["embs"]))
    return {}


def save_db(db):
    if db:
        names, embs = zip(*db.items())
        np.savez(DB_PATH, names=np.array(names), embs=np.array(embs))


face_db = load_db()


# ---------- core logic ----------
def analyse_frame(img):
    """Detect faces, recognise known ones, return annotated img + JSON results."""
    if img is None:  # guard for empty webcam frame
        return None, ""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = DeepFace.analyze(
        rgb, actions=["emotion"], detector_backend="mtcnn"
    )  # list of dicts
    if not isinstance(results, list):
        results = [results]

    output_json = []
    for res in results:
        x, y, w, h = [int(res["region"][k]) for k in ("x", "y", "w", "h")]
        face_crop = rgb[y : y + h, x : x + w]
        emb = DeepFace.represent(face_crop, detector_backend="mtcnn")[0]["embedding"]

        # recognise
        label, best = "Unknown", 1.0
        for name, db_emb in face_db.items():
            dist = cosine(emb, db_emb)
            if dist < best:
                best = dist
                label = name
        if best > 0.4:  # threshold: lower => stricter match
            label = "Unknown"

        # draw rectangle + label
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{label} | {res['dominant_emotion']}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        output_json.append(
            {
                "name": label,
                "bbox": {"x": x, "y": y, "w": w, "h": h},
                "emotion": res["dominant_emotion"],
                "similarity": round(1 - best, 3),
            }
        )

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), json.dumps(output_json, indent=2)

def add_face(img, person_name):
    """Save a single face embedding under the given name."""
    if img is None or not person_name:
        return "Need both image and name!"

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    res = DeepFace.analyze(rgb, actions=[], detector_backend="mtcnn")
    # nếu chỉ một mặt, DeepFace trả về dict; nếu nhiều mặt -> list
    if isinstance(res, list):
        res = res[0]

    x, y, w, h = [int(res["region"][k]) for k in ("x", "y", "w", "h")]
    face_crop = rgb[y:y+h, x:x+w]
    emb = DeepFace.represent(face_crop, detector_backend="mtcnn")[0]["embedding"]

    face_db[person_name] = np.asarray(emb, dtype=np.float32)
    save_db(face_db)
    return f"Saved ‘{person_name}’ with {len(face_db)} total faces."


# ---------- build Gradio UI ----------
with gr.Blocks(title="DeepFace demo") as demo:
    gr.Markdown(
        "## Real-time face detection, recognition and emotion\n"
        "Upload a photo **or** switch to *webcam*."
    )
    with gr.Tab("Recognise"):
        inp = gr.Image(sources=["upload", "webcam"], type="numpy", streaming=True)
        out_img = gr.Image()
        out_json = gr.Textbox(label="Results (JSON)")
        inp.stream(analyse_frame, inp, [out_img, out_json])

    with gr.Tab("Add new face"):
        add_img = gr.Image(type="numpy", sources=["upload"])  # <— đổi here
        name_box = gr.Textbox(label="Person name")
        add_btn = gr.Button("Save to DB")
        info = gr.Textbox(label="Status")
        add_btn.click(add_face, [add_img, name_box], info)

demo.launch()
