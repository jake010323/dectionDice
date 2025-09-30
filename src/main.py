import logging
import os
import sys
import time
import json
import cv2
import ctypes
import threading
import tkinter as tk
import requests
from PIL import Image, ImageTk
from collections import defaultdict
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

# Set Windows app ID for proper icon on taskbar
if os.name == 'nt':
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(u"PiztechSiboApp")

def resource_path(filename):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), filename)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load config.json: {e}")
    return {}

def save_config(data):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)
        logging.info("✅ config.json saved.")
    except Exception as e:
        logging.error(f"Failed to save config.json: {e}")

try:
    import scipy._lib._util
except ImportError as e:
    logging.error(f"SciPy module missing: {e}")
    raise

from inference.core.interfaces.stream.inference_pipeline import InferencePipeline

# Setup logging
logging.basicConfig(
    filename='app_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.debug('Script started')

# Config
os.environ["PALIGEMMA_ENABLED"] = "False"
output_dir = "saved_frames"
os.makedirs(output_dir, exist_ok=True)

result_file_path = os.path.join(os.path.dirname(__file__), "data", "latest_result.json")
os.makedirs(os.path.dirname(result_file_path), exist_ok=True)

# Load config
config = load_config()
api_url = config.get("api_url", "https://your-api.com/current-match")

# Globals
latest_frame = None
exit_requested = False
detection_text = ""
frame_results = []
frame_results_lock = threading.Lock()
current_match_id = None

def list_video_devices(max_devices=10):
    logging.debug("Listing available video devices")
    available_devices = []
    for device_id in range(max_devices):
        cap = cv2.VideoCapture(device_id)
        if cap.isOpened():
            available_devices.append(device_id)
            cap.release()
    return available_devices

def fetch_current_match_loop():
    global current_match_id
    while not exit_requested:
        try:
            response = requests.get(api_url)
            if response.status_code == 200:
                data = response.json()
                # Extract reference_id from the nested "data" object
                current_match_id = data.get("data", {}).get("reference_id")
                logging.debug(f"Fetched match reference ID: {current_match_id}")
            else:
                logging.warning(f"Failed to fetch match ID: {response.status_code}")
        except Exception as e:
            logging.error(f"Error fetching match ID: {e}")
        time.sleep(2)


def display_loop_tkinter(window):
    global latest_frame, exit_requested, detection_text

    icon_path = resource_path("icon.ico")
    try:
        window.iconbitmap(icon_path)
    except Exception as e:
        logging.warning(f"Could not set icon.ico: {e}")

    canvas = tk.Canvas(window, width=1280, height=720)
    canvas.pack()

    def update_frame():
        if exit_requested:
            window.destroy()
            return

        if latest_frame is not None:
            try:
                frame_resized = cv2.resize(latest_frame, (1280, 720))
                if detection_text:
                    cv2.putText(
                        frame_resized, detection_text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA
                    )
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                canvas.imgtk = imgtk
                canvas.create_image(0, 0, anchor='nw', image=imgtk)
            except Exception as e:
                logging.error(f"Tkinter display error: {e}")

        window.after(50, update_frame)

    def on_close():
        global exit_requested
        logging.info("Window closed")
        exit_requested = True
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", on_close)
    window.title("Piztech - Sibo999")
    window.geometry("1280x720")
    update_frame()

def on_prediction(prediction_result, video_frame):
    global latest_frame, exit_requested

    if exit_requested:
        return

    predictions = prediction_result.get('predictions', [])
    image = video_frame.image
    latest_frame = image.copy()
    timestamp = time.time()

    class_values = []
    for item in predictions:
        try:
            class_name = item[5]["class_name"]
            class_value = int(class_name)
            class_values.append(class_value)
        except Exception as e:
            logging.warning(f"Error reading class_name: {e}")

    if len(class_values) == 3:
        breakdown = " + ".join(map(str, class_values))
        total = sum(class_values)
        with frame_results_lock:
            frame_results.append({
                "timestamp": timestamp,
                "values": tuple(sorted(class_values)),
                "breakdown": breakdown,
                "total": total,
                "image": image.copy()
            })

def pick_least_frequent_result_loop():
    global detection_text

    while not exit_requested:
        time.sleep(1.0)
        now = time.time()

        with frame_results_lock:
            recent = [r for r in frame_results if now - r["timestamp"] <= 1.0]
            frame_results.clear()

        if not recent:
            continue

        count_map = defaultdict(list)
        for r in recent:
            count_map[r["values"]].append(r)

        rarest = min(count_map.items(), key=lambda item: len(item[1]))
        chosen_result = rarest[1][0]
        detection_text = f"{chosen_result['breakdown']} = {chosen_result['total']}"

        try:
            if os.path.exists(result_file_path):
                with open(result_file_path, "r") as f:
                    results = json.load(f)
            else:
                results = []

            if any(entry.get("currentMatchId") == current_match_id for entry in results):
                logging.info(f"Match ID {current_match_id} already recorded. Skipping append.")
            else:
                results.append({
                    "timestamp": chosen_result["timestamp"],
                    "dices": list(chosen_result["values"]),
                    "total": chosen_result["total"],
                    "currentMatchId": current_match_id
                })
                with open(result_file_path, "w") as f:
                    json.dump(results, f, indent=2)
                logging.info(f"✅ Saved result for match ID: {current_match_id}")

        except Exception as e:
            logging.error(f"Failed to write result file: {e}")

        local_time = time.localtime(chosen_result["timestamp"])
        ms = int((chosen_result["timestamp"] - int(chosen_result["timestamp"])) * 1000)
        formatted_time = time.strftime("%d/%m/%y %H:%M:%S", local_time) + f".{ms:03d}"
        logging.info(f"[{formatted_time}] ✅ Selected rarest result: {detection_text}")
        print(f"[{formatted_time}] ✅ Selected rarest result: {detection_text}")

        def clear_text():
            global detection_text
            time.sleep(2)
            detection_text = ""

        threading.Thread(target=clear_text, daemon=True).start()

def start_pipeline(video_index):
    try:
        pipeline = InferencePipeline.init_with_workflow(
            workflow_id="detect-count-and-visualize-4",
            workspace_name="dices-g4rgf",
            api_key="qKXfmUY6GQgT6lxG4EbL",
            video_reference=video_index,
            max_fps=10,
            workflows_thread_pool_workers=2,
            on_prediction=on_prediction,
            video_source_properties={
                "frame_width": 1280.0,
                "frame_height": 720.0,
                "fps": 50.0
            },
            workflows_parameters={},
            workflow_init_parameters={},
            use_workflow_definition_cache=True,
            serialize_results=False
        )
        pipeline.start()
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

def launch_main_window(video_index):
    global exit_requested
    exit_requested = False
    root = tk.Tk()
    threading.Thread(target=fetch_current_match_loop, daemon=True).start()
    threading.Thread(target=start_pipeline, args=(video_index,), daemon=True).start()
    threading.Thread(target=pick_least_frequent_result_loop, daemon=True).start()
    display_loop_tkinter(root)
    root.mainloop()

def show_camera_selector(devices):
    global api_url
    root = tk.Tk()
    root.title("Chọn Camera")
    root.geometry("300x400")

    icon_path = resource_path("icon.ico")
    try:
        root.iconbitmap(icon_path)
    except Exception as e:
        logging.warning(f"Could not set icon.ico: {e}")

    # Click counter
    click_count = 0

    # Title label
    title_label = tk.Label(root, text="Chọn Thiết Bị", font=("Arial", 14))
    title_label.pack(pady=10)

    # API input frame (hidden initially)
    api_frame = tk.Frame(root)
    api_label = tk.Label(api_frame, text="Nhập API URL:", font=("Arial", 10))
    url_var = tk.StringVar(value=api_url)
    url_entry = tk.Entry(api_frame, textvariable=url_var, width=40)

    def reveal_api_input(event=None):
        nonlocal click_count
        click_count += 1
        if click_count == 3:
            api_label.pack()
            url_entry.pack(pady=5)
            api_frame.pack(pady=10)

    title_label.bind("<Button-1>", reveal_api_input)

    def on_close():
        global exit_requested
        exit_requested = True
        logging.info("Selector closed")
        root.after(100, lambda: os._exit(0))

    root.protocol("WM_DELETE_WINDOW", on_close)

    def select_camera(dev_index):
        global api_url
        api_url = url_var.get().strip()
        save_config({"api_url": api_url})
        logging.info(f"Using API URL: {api_url}")
        root.destroy()
        launch_main_window(dev_index)

    for dev in devices:
        tk.Button(
            root,
            text=f"Camera {dev}",
            font=("Arial", 12),
            command=lambda d=dev: select_camera(d)
        ).pack(pady=5, ipadx=10, ipady=5)

    root.mainloop()

def start_local_server():
    app = Flask(__name__)
    CORS(app)

    @app.route('/last_result.json')
    def serve_result():
        directory = os.path.dirname(result_file_path)
        filename = os.path.basename(result_file_path)
        full_path = os.path.join(directory, filename)

        if not os.path.exists(full_path):
            return jsonify({"error": "File not found", "path": full_path}), 404

        print(f"[Flask] Serving: {full_path}")
        return send_from_directory(directory, filename)

    app.run(host='0.0.0.0', port=9999, debug=False)

if __name__ == "__main__":
    logging.debug("App starting...")
    threading.Thread(target=start_local_server, daemon=True).start()
    devices = list_video_devices()
    if not devices:
        logging.error("No video devices found. Exiting.")
        sys.exit(1)
    show_camera_selector(devices)
