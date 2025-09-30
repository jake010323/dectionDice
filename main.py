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
from flask import Flask, jsonify, Response
from flask_cors import CORS
import pathlib
from urllib.parse import urlparse, parse_qs
import queue
import obsws_python as obs
from inference.core.interfaces.stream.inference_pipeline import InferencePipeline

if os.name == 'nt':
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(u"PiztechSiboApp")

def resource_path(filename):
    """Get path for frozen or dev mode"""
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), filename)

def get_app_dir():
    if getattr(sys, 'frozen', False):
        base = pathlib.Path(os.getenv("APPDATA", pathlib.Path.home())) if os.name == "nt" else pathlib.Path.home() / ".config"
        return base / "PiztechSiboApp"
    return pathlib.Path(__file__).parent.resolve()

APP_DIR = get_app_dir()
APP_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = APP_DIR / "config.json"
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
RESULT_FILE_PATH = DATA_DIR / "latest_result.json"
if not RESULT_FILE_PATH.exists():
    with open(RESULT_FILE_PATH, "w") as f:
        json.dump([], f)

def load_config():
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            pass
    return {}

def save_config(data):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        pass

config = load_config()
api_url = config.get("api_url", "")
workflow_id = config.get("workflow_id", "")
OBS_PASSWORD = config.get("obs_password", "")

latest_frame = None
latest_frame_lock = threading.Lock()
exit_requested = False
frame_queue = queue.Queue(maxsize=30)
saver_queue = queue.Queue()
reference_id = None
start_time = ""
detection_time = 1


# ======================================================
# Helper Functions
# ======================================================
def get_gb_from_url(url):
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        return params.get("gb", [""])[0]
    except Exception:
        return ""

def list_video_devices(max_devices=10):
    devices = []
    api = cv2.CAP_DSHOW if os.name == "nt" else 0
    for i in range(max_devices):
        try:
            cap = cv2.VideoCapture(i, api)
            if not cap.isOpened():
                cap.release()
                continue
            ret, _ = cap.read()
            if ret:
                devices.append(i)
            cap.release()
        except Exception:
            try: cap.release()
            except: pass
            continue
    return devices

# ======================================================
# OBS Control
# ======================================================
OBS_HOST = "localhost"
OBS_PORT = 4455
obs_client = None
last_scene = None

def connect_obs():
    global obs_client
    try:
        obs_client = obs.ReqClient(host=OBS_HOST, port=OBS_PORT, password=OBS_PASSWORD)
    except Exception as e:
        pass

def switch_obs_scene(scene_name):
    global obs_client, last_scene
    if scene_name == last_scene:
        return
    last_scene = scene_name
    if obs_client is None:
        connect_obs()
    if obs_client:
        try:
            obs_client.set_current_program_scene(scene_name)
        except Exception as e:
            pass

# ======================================================
# Current Match Loop
# ======================================================
def fetch_current_match_loop():
    global reference_id, start_time
    empty_entry_sent = False 

    while not exit_requested:
        try:
            response = requests.get(api_url, timeout=2)
            if response.status_code == 200:
                data = response.json().get("data", {})
                ref_id = data.get("reference_id")
                st_time = data.get("start_time", "")
                status = data.get("status", 1)
                if ref_id and st_time:
                    reference_id = ref_id
                    start_time = st_time

                if status == 0:
                    switch_obs_scene("close")
                    empty_entry_sent = False 
                else:
                    switch_obs_scene("open")
                    if not empty_entry_sent:
                        empty_entry = {
                            "gb": get_gb_from_url(api_url),
                            "timestamp": '',
                            "dices": [],
                            "total": 0,
                            "reference_id": reference_id,
                            "start_time": start_time,
                            "status": ""
                        }
                        saver_queue.put(empty_entry)
                        empty_entry_sent = True

        except Exception as e:
            pass
        time.sleep(0.1)


# ======================================================
# Prediction Handler
# ======================================================
def on_prediction(prediction_result, video_frame):
    global latest_frame
    with latest_frame_lock:
        latest_frame = video_frame.image.copy() 
    gray_frame = cv2.cvtColor(video_frame.image, cv2.COLOR_BGR2GRAY)
    try:
        frame_queue.put_nowait((time.time(), gray_frame, prediction_result))
    except queue.Full:
        try:
            frame_queue.get_nowait()
            frame_queue.put_nowait((time.time(), gray_frame, prediction_result))
        except queue.Empty:
            pass



# ======================================================
# Frame Worker
# ======================================================
LOCAL_IMAGE_DIR = DATA_DIR / "payout_images"
LOCAL_IMAGE_DIR.mkdir(exist_ok=True)

def frame_worker():
    last_saved_values = None
    buffer = []
    while not exit_requested:
        try:
            timestamp, frame, prediction_result = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        predictions = prediction_result.get('predictions', [])
        class_values = []
        for item in predictions:
            try:
                class_name = item[5]["class_name"]
                if str(class_name).isdigit():
                    class_values.append(int(class_name))
            except Exception:
                continue

        if len(class_values) != 3:
            continue

        class_values_sorted = tuple(sorted(class_values))
        buffer.append({"timestamp": timestamp, "values": class_values_sorted, "frame": frame})

        now = time.time()
        buffer = [b for b in buffer if now - b["timestamp"] <= 2]
        if not buffer:
            continue

        count_map = defaultdict(list)
        for b in buffer:
            count_map[b["values"]].append(b)
        most_freq = max(count_map.items(), key=lambda x: len(x[1]))
        chosen = most_freq[1][0]

        if chosen["values"] != last_saved_values:
            last_saved_values = chosen["values"]
            total = sum(chosen["values"])
            gb_value = get_gb_from_url(api_url)
            entry = {
                "gb": gb_value,
                "timestamp": chosen["timestamp"],
                "dices": list(chosen["values"]),
                "total": total,
                "status": "payout",
                "reference_id": reference_id,
                "start_time": start_time
            }

            try:
                dice_str = "-".join(str(v) for v in chosen["values"])
                img_filename = LOCAL_IMAGE_DIR / f"{int(time.time() * 1000)}_{dice_str}.png"
                cv2.imwrite(str(img_filename), chosen["frame"])
            except Exception as e:
                print("Error saving payout image:", e)

            def delayed_file_save(entry):
                time.sleep(1.5)
                saver_queue.put(entry)
            threading.Thread(target=delayed_file_save, args=(entry,), daemon=True).start()


# ======================================================
# Result Saver Worker
# ======================================================
def result_saver_worker():
    last_entry = None
    while not exit_requested:
        try:
            while True:
                last_entry = saver_queue.get_nowait()
        except queue.Empty:
            pass

        if last_entry:
            try:
                with open(RESULT_FILE_PATH, "w") as f:
                    json.dump([last_entry], f, indent=2)
                last_entry = None
            except Exception as e:
                pass
        time.sleep(0.1)

# ======================================================
# OpenCV Display Loop
# ======================================================
def display_loop_opencv(width=1280, height=720, desired_fps=30):
    global latest_frame, exit_requested
    window_name = "Piztech - Sibo999"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    frame_interval = 1.0 / max(1, desired_fps)

    while not exit_requested:
        t0 = time.time()
        with latest_frame_lock:
            lf = latest_frame.copy() if (latest_frame is not None and hasattr(latest_frame, "copy")) else None

        if lf is not None:
            try:
                if lf.shape[1] != width or lf.shape[0] != height:
                    lf = cv2.resize(lf, (width, height))
                cv2.imshow(window_name, lf)
            except Exception as e:
                pass

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            exit_requested = True
            break
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                exit_requested = True
                break
        except Exception:
            exit_requested = True
            break

        elapsed = time.time() - t0
        time.sleep(max(0, frame_interval - elapsed))

    cv2.destroyAllWindows()
    os._exit(0)

# ======================================================
# Start Inference Pipeline
# ======================================================
def start_pipeline(video_index):
    try:
        pipeline = InferencePipeline.init_with_workflow(
            workflow_id=workflow_id,
            workspace_name="dices-g4rgf",
            api_key="qKXfmUY6GQgT6lxG4EbL",
            video_reference=video_index,
            max_fps=50,
            workflows_thread_pool_workers=12,
            on_prediction=on_prediction,
            video_source_properties={
                "frame_width": 640.0,
                "frame_height": 320.0,
                "fps": 50.0
            },
            workflows_parameters={},
            workflow_init_parameters={},
            use_workflow_definition_cache=False,
            serialize_results=False
        )
        pipeline.start()
    except Exception as e:
        pass

# ======================================================
# Camera Selector UI
# ======================================================
def show_camera_selector(devices):
    global api_url, workflow_id, OBS_PASSWORD
    root = tk.Tk()
    root.title("Chọn Camera & Config")
    root.geometry("350x550")
    icon_path = resource_path("icon.ico")
    try:
        root.iconbitmap(icon_path)
    except:
        pass
    click_count = 0
    title_label = tk.Label(root, text="Chọn Thiết Bị", font=("Arial", 14))
    title_label.pack(pady=10)

    api_frame = tk.Frame(root)
    api_label = tk.Label(api_frame, text="Nhập API URL:", font=("Arial", 10))
    url_var = tk.StringVar(value=api_url)
    url_entry = tk.Entry(api_frame, textvariable=url_var, width=40)

    workflow_frame = tk.Frame(root)
    workflow_label = tk.Label(workflow_frame, text="Nhập Workflow ID:", font=("Arial", 10))
    workflow_var = tk.StringVar(value=workflow_id)
    workflow_entry = tk.Entry(workflow_frame, textvariable=workflow_var, width=40)

    obs_frame = tk.Frame(root)
    obs_label = tk.Label(obs_frame, text="Nhập OBS Password:", font=("Arial", 10))
    obs_var = tk.StringVar(value=OBS_PASSWORD)
    obs_entry = tk.Entry(obs_frame, textvariable=obs_var, width=40, show="*")

    def reveal_hidden_inputs(event=None):
        nonlocal click_count
        click_count += 1
        if click_count == 3:
            api_label.pack()
            url_entry.pack(pady=5)
            api_frame.pack(pady=10)
            workflow_label.pack()
            workflow_entry.pack(pady=5)
            workflow_frame.pack(pady=10)
            obs_label.pack()
            obs_entry.pack(pady=5)
            obs_frame.pack(pady=10)

    title_label.bind("<Button-1>", reveal_hidden_inputs)

    def on_close():
        global exit_requested
        exit_requested = True
        root.after(100, lambda: os._exit(0))

    root.protocol("WM_DELETE_WINDOW", on_close)

    def select_camera(dev_index):
        global api_url, workflow_id, OBS_PASSWORD
        api_url = url_var.get().strip()
        workflow_id = workflow_var.get().strip()
        OBS_PASSWORD = obs_var.get().strip()
        save_config({"api_url": api_url, "workflow_id": workflow_id, "obs_password": OBS_PASSWORD})
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

# ======================================================
# Local Server
# ======================================================
def start_local_server():
    app = Flask(__name__)
    CORS(app, resources={
        r"/*": {
            "origins": [
                r"http://127\\.0\\.0\\.1(:\\d+)?",
                r"http://localhost(:\\d+)?",
                r"https://adop.gal999.com"
            ]
        }
    })
    @app.route('/last_result.json')
    def serve_result():
        try:
            if not RESULT_FILE_PATH.exists():
                return jsonify([])
            with open(RESULT_FILE_PATH, "r") as f:
                try:
                    data = json.load(f)
                except:
                    data = []
            return jsonify(data)
        except Exception as e:
            return jsonify([])
    app.run(host='127.0.0.1', port=9999, debug=False, threaded=True)

# ======================================================
# Launch Main Window
# ======================================================
def launch_main_window(video_index):
    global exit_requested
    exit_requested = False
    threading.Thread(target=fetch_current_match_loop, daemon=True).start()
    threading.Thread(target=frame_worker, daemon=True).start()
    threading.Thread(target=result_saver_worker, daemon=True).start()
    threading.Thread(target=start_pipeline, args=(video_index,), daemon=True).start()
    display_loop_opencv()

# ======================================================
# Main Entry
# ======================================================
if __name__ == "__main__":
    threading.Thread(target=start_local_server, daemon=True).start()
    devices = list_video_devices()
    if not devices:
        sys.exit(1)
    show_camera_selector(devices)