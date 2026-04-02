"""
Servidor Flask — API para o Video Cutter.
Recebe uploads de vídeo, processa e retorna os clips cortados.
"""

import os
import uuid
import json
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from video_cutter import VideoCutter

app = Flask(__name__, static_folder="public", static_url_path="")
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "m4v"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

cutter = VideoCutter(output_dir=OUTPUT_FOLDER)

# Estado dos processamentos
jobs = {}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return send_from_directory("public", "index.html")


@app.route("/api/upload", methods=["POST"])
def upload_video():
    """Recebe upload do vídeo."""
    if "video" not in request.files:
        return jsonify({"error": "Nenhum vídeo enviado"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Nenhum arquivo selecionado"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Formato não suportado. Use: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    # Salvar com nome único
    job_id = str(uuid.uuid4())[:8]
    ext = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{job_id}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    jobs[job_id] = {
        "id": job_id,
        "filename": file.filename,
        "filepath": filepath,
        "status": "uploaded"
    }

    return jsonify({
        "job_id": job_id,
        "filename": file.filename,
        "message": "Vídeo enviado com sucesso"
    })


@app.route("/api/process/<job_id>", methods=["POST"])
def process_video(job_id):
    """Processa o vídeo e gera os clips."""
    if job_id not in jobs:
        return jsonify({"error": "Job não encontrado"}), 404

    job = jobs[job_id]

    # Parâmetros do request
    data = request.get_json() or {}
    num_clips = min(int(data.get("num_clips", 5)), 20)
    clip_duration = min(int(data.get("clip_duration", 30)), 60)
    clip_duration = max(clip_duration, 15)
    vertical = data.get("vertical", True)

    job["status"] = "processing"

    try:
        results = cutter.process_video(
            job["filepath"],
            clip_duration=clip_duration,
            num_clips=num_clips,
            vertical=vertical
        )

        job["status"] = "completed"
        job["results"] = results

        # Preparar resposta com URLs dos clips
        clips_info = []
        for clip in results["clips"]:
            clip_filename = os.path.basename(clip["path"])
            video_name = os.path.splitext(os.path.basename(job["filepath"]))[0]
            clips_info.append({
                "url": f"/api/clip/{video_name}/{clip_filename}",
                "start": clip["start"],
                "end": clip["end"],
                "duration": clip["duration"],
                "energy_score": clip["energy_score"]
            })

        return jsonify({
            "job_id": job_id,
            "status": "completed",
            "total_clips": len(clips_info),
            "clips": clips_info
        })

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/clip/<video_name>/<clip_name>")
def get_clip(video_name, clip_name):
    """Retorna um clip processado."""
    clip_path = os.path.join(OUTPUT_FOLDER, video_name, clip_name)
    if not os.path.exists(clip_path):
        return jsonify({"error": "Clip não encontrado"}), 404
    return send_file(clip_path, mimetype="video/mp4")


@app.route("/api/status/<job_id>")
def get_status(job_id):
    """Retorna o status do processamento."""
    if job_id not in jobs:
        return jsonify({"error": "Job não encontrado"}), 404
    job = jobs[job_id]
    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "filename": job["filename"]
    })


if __name__ == "__main__":
    print("=== Video Cutter Server ===")
    print("Acesse: http://localhost:5000")
    print("===========================")
    app.run(debug=True, port=5000)
