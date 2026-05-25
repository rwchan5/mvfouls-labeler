import json
import csv
import re
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, abort, redirect

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "SoccerNetData" / "mvfouls"
LABELS_FILE = BASE_DIR / "box_labels.csv"
SPLITS = ["train", "valid", "test"]

CAM_PRIORITY = {
    "Main camera center": 0,
    "Main camera left": 1,
    "Main camera right": 2,
    "Main behind the goal": 3,
    "Spider camera": 4,
    "Close-up player or field referee": 5,
    "Close-up behind the goal": 6,
    "Close-up corner": 7,
    "Goal line technology camera": 8,
    "Inside the goal": 9,
    "Other": 10,
    "Close-up side staff": 11,
}

SEVERITY_LABELS = {
    "1.0": "No card",
    "2.0": "No card (border)",
    "3.0": "Yellow card",
    "4.0": "Red card (border)",
    "5.0": "Red card",
}

CSV_FIELDS = ["split", "action_id", "box_location", "unclear_reason", "notes", "labeled_at"]


def load_actions():
    actions = []
    for split in SPLITS:
        ann_path = DATA_DIR / split / "annotations.json"
        if not ann_path.exists():
            continue
        with open(ann_path) as f:
            data = json.load(f)
        for key, action in data["Actions"].items():
            num = int(key)
            action_id = f"action_{num}"
            clips_sorted = sorted(
                action.get("Clips", []),
                key=lambda c: CAM_PRIORITY.get(c.get("Camera type", ""), 99),
            )
            clips = []
            for clip in clips_sorted:
                url_parts = clip.get("Url", "").replace("\\", "/").split("/")
                clip_name = url_parts[-1] if url_parts else f"clip_0"
                clips.append({
                    "url": f"/video/{split}/{action_id}/{clip_name}.mp4",
                    "camera_type": clip.get("Camera type", "Unknown"),
                    "replay_speed": clip.get("Replay speed", 1.0),
                })
            url_local = action.get("UrlLocal", "")
            match_name = url_local.replace("\\", "/").split("/")[-1] if url_local else "Unknown"
            actions.append({
                "split": split,
                "action_id": action_id,
                "action_num": num,
                "clips": clips,
                "offence": action.get("Offence", ""),
                "severity": action.get("Severity", ""),
                "severity_label": SEVERITY_LABELS.get(action.get("Severity", ""), "N/A"),
                "action_class": action.get("Action class", ""),
                "bodypart": action.get("Bodypart", ""),
                "contact": action.get("Contact", ""),
                "handball": action.get("Handball", ""),
                "match": match_name,
                "try_to_play": action.get("Try to play", ""),
                "touch_ball": action.get("Touch ball", ""),
            })
    split_order = {s: i for i, s in enumerate(SPLITS)}
    actions.sort(key=lambda a: (split_order[a["split"]], a["action_num"]))
    return actions


def load_labels():
    labels = {}
    if LABELS_FILE.exists():
        with open(LABELS_FILE, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["split"], row["action_id"])
                labels[key] = {
                    "box_location": row.get("box_location", ""),
                    "unclear_reason": row.get("unclear_reason", ""),
                    "notes": row.get("notes", ""),
                    "labeled_at": float(row.get("labeled_at", 0) or 0),
                }
    return labels


def save_label(split, action_id, box_location, unclear_reason="", notes=""):
    labels = load_labels()
    labels[(split, action_id)] = {
        "box_location": box_location,
        "unclear_reason": unclear_reason,
        "notes": notes,
        "labeled_at": time.time(),
    }
    with open(LABELS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for (s, a), v in labels.items():
            writer.writerow({
                "split": s,
                "action_id": a,
                "box_location": v["box_location"],
                "unclear_reason": v["unclear_reason"],
                "notes": v["notes"],
                "labeled_at": v["labeled_at"],
            })


def compute_eta(labels, total):
    timestamps = sorted(v["labeled_at"] for v in labels.values() if v["labeled_at"] > 0)
    if len(timestamps) < 2:
        return None
    recent = timestamps[-21:]
    intervals = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    avg_sec = sum(intervals) / len(intervals)
    if avg_sec <= 0:
        return None
    remaining = total - len(labels)
    return avg_sec * remaining


ACTIONS = load_actions()
ACTION_INDEX = {(a["split"], a["action_id"]): i for i, a in enumerate(ACTIONS)}


@app.route("/")
def overview():
    labels = load_labels()
    counts = {"inside": 0, "outside": 0, "unclear": 0, "unlabeled": 0}
    splits_data = {s: {"actions": [], "inside": 0, "outside": 0, "unclear": 0, "unlabeled": 0, "total": 0} for s in SPLITS}

    for action in ACTIONS:
        key = (action["split"], action["action_id"])
        label_info = labels.get(key)
        loc = label_info["box_location"] if label_info else None
        status = loc if loc else "unlabeled"
        counts[status] += 1
        splits_data[action["split"]][status] += 1
        splits_data[action["split"]]["total"] += 1
        splits_data[action["split"]]["actions"].append({
            "split": action["split"],
            "action_id": action["action_id"],
            "action_num": action["action_num"],
            "label": loc,
            "action_class": action["action_class"],
            "offence": action["offence"],
        })

    labeled_count = counts["inside"] + counts["outside"] + counts["unclear"]
    eta_seconds = compute_eta(labels, len(ACTIONS))
    eta_str = None
    if eta_seconds:
        h = int(eta_seconds // 3600)
        m = int((eta_seconds % 3600) // 60)
        eta_str = f"{h}h {m}m remaining" if h > 0 else f"{m}m remaining"

    return render_template(
        "overview.html",
        counts=counts,
        labeled_count=labeled_count,
        total=len(ACTIONS),
        splits_data=splits_data,
        splits=SPLITS,
        eta_str=eta_str,
    )


@app.route("/label/continue")
def label_continue():
    labels = load_labels()
    for action in ACTIONS:
        if (action["split"], action["action_id"]) not in labels:
            return redirect(f"/label/{action['split']}/{action['action_id']}")
    return redirect("/")


@app.route("/label/<split>/<action_id>")
def label_page(split, action_id):
    labels = load_labels()
    key = (split, action_id)
    idx = ACTION_INDEX.get(key)
    if idx is None:
        abort(404)

    action = ACTIONS[idx]
    current_label = labels.get(key)
    prev_action = ACTIONS[idx - 1] if idx > 0 else None
    next_action = ACTIONS[idx + 1] if idx < len(ACTIONS) - 1 else None

    next_unlabeled = None
    for i in range(idx + 1, len(ACTIONS)):
        a = ACTIONS[i]
        if (a["split"], a["action_id"]) not in labels:
            next_unlabeled = a
            break
    if next_unlabeled is None:
        for i in range(0, idx):
            a = ACTIONS[i]
            if (a["split"], a["action_id"]) not in labels:
                next_unlabeled = a
                break

    return render_template(
        "label.html",
        action=action,
        idx=idx,
        total=len(ACTIONS),
        current_label=current_label,
        prev_action=prev_action,
        next_action=next_action,
        next_unlabeled=next_unlabeled,
    )


@app.route("/api/label", methods=["POST"])
def api_label():
    data = request.get_json()
    split = data.get("split")
    action_id = data.get("action_id")
    box_location = data.get("box_location")
    unclear_reason = data.get("unclear_reason", "")
    notes = data.get("notes", "")

    if not all([split, action_id, box_location]):
        return jsonify({"error": "Missing fields"}), 400
    if box_location not in ("inside", "outside", "unclear"):
        return jsonify({"error": "Invalid label"}), 400

    save_label(split, action_id, box_location, unclear_reason, notes)

    idx = ACTION_INDEX.get((split, action_id))
    next_action = ACTIONS[idx + 1] if idx is not None and idx < len(ACTIONS) - 1 else None
    next_url = f"/label/{next_action['split']}/{next_action['action_id']}" if next_action else None

    return jsonify({"success": True, "next_url": next_url})


@app.route("/video/<split>/<action_id>/<filename>")
def serve_video(split, action_id, filename):
    if not re.match(r"^[\w\-]+\.mp4$", filename):
        abort(400)
    video_path = DATA_DIR / split / action_id / filename
    if not video_path.exists():
        abort(404)
    return send_file(str(video_path), mimetype="video/mp4")


if __name__ == "__main__":
    print(f"Loaded {len(ACTIONS)} actions across {len(SPLITS)} splits")
    print("Open http://localhost:5050")
    app.run(debug=False, port=5050)
