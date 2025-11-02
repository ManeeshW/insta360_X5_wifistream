import zenoh
import cv2
import numpy as np
import json
import threading
import time
import configparser
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QFormLayout, QDoubleSpinBox, QSpinBox, QDialogButtonBox, QDialog, QPushButton, QWidget
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer

# Zenoh configuration for UDP multicast scouting (no specific IPs needed)
# Using the same custom port (7450) as Node 1
# Set the multicast interface to 'auto' (default, but explicitly set for clarity)
# If this doesn't work, replace 'auto' with your network interface name (e.g., 'wlan0' or 'enp0s3')
# To find your interface, run 'ip link show' in terminal and look for the active network device (not 'lo')
conf = zenoh.Config()
conf.insert_json5("scouting/multicast/enabled", "true")
conf.insert_json5("scouting/multicast/address", '"224.0.0.224:7450"')
conf.insert_json5("scouting/multicast/interface", '"auto"')
conf.insert_json5("listen/endpoints", '["udp/0.0.0.0:0"]')  # Auto-assign unicast port

session = zenoh.open(conf)

print(f"Node 2 Zenoh ID: {session.info.zid()}")  # For debugging

# Default view parameters
defaults = {
    "forward": {"enable": True, "fov_x": 82.0, "fov_y": 82.0, "width": 640, "height": 480, "quality": 100},
    "rear": {"enable": False, "fov_x": 90.0, "fov_y": 90.0, "width": 640, "height": 480, "quality": 100},
    "left": {"enable": False, "fov_x": 90.0, "fov_y": 90.0, "width": 640, "height": 480, "quality": 100},
    "right": {"enable": False, "fov_x": 90.0, "fov_y": 90.0, "width": 640, "height": 480, "quality": 100},
    "downward": {"enable": False, "fov_x": 120.0, "fov_y": 120.0, "width": 480, "height": 640, "quality": 100}
}

# View keys
view_keys = {
    "forward": "cam_forward",
    "rear": "cam_rear",
    "left": "cam_left",
    "right": "cam_right",
    "downward": "cam_downward"
}

# Shared parameters class
class SharedParams:
    def __init__(self, config):
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.views = {}
        for view in view_keys:
            sec = view
            if config.has_section(sec):
                self.views[view] = {
                    "enable": config.getboolean(sec, "enable", fallback=defaults[view]["enable"]),
                    "fov_x": config.getfloat(sec, "fov_x", fallback=defaults[view]["fov_x"]),
                    "fov_y": config.getfloat(sec, "fov_y", fallback=defaults[view]["fov_y"]),
                    "width": config.getint(sec, "width", fallback=defaults[view]["width"]),
                    "height": config.getint(sec, "height", fallback=defaults[view]["height"]),
                    "quality": config.getint(sec, "quality", fallback=defaults[view]["quality"])
                }
            else:
                self.views[view] = defaults[view].copy()

    def send(self, pub):
        data = {
            "global_yaw": self.yaw,
            "global_pitch": self.pitch,
            "global_roll": self.roll,
            "views": {v: p.copy() for v, p in self.views.items()}
        }
        pub.put(json.dumps(data).encode(), encoding=zenoh.Encoding.APPLICATION_JSON)

    def save_config(self):
        config = configparser.ConfigParser()
        for view, params in self.views.items():
            config[view] = {
                "enable": str(params["enable"]),
                "fov_x": str(params["fov_x"]),
                "fov_y": str(params["fov_y"]),
                "width": str(params["width"]),
                "height": str(params["height"]),
                "quality": str(params["quality"])
            }
        with open("gui_config.cfg", "w") as configfile:
            config.write(configfile)

# View Window class
class ViewWindow(QMainWindow):
    def __init__(self, name, session, shared_params, view_keys):
        super().__init__()
        self.name = name
        self.shared_params = shared_params
        self.setWindowTitle(f"{name.capitalize()} View")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()

        self.label = QLabel(self)
        self.layout.addWidget(self.label)

        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.open_settings)
        self.layout.addWidget(self.settings_button)

        self.pressed_keys = set()

        if name == "forward":
            self.is_recording = False
            self.record_button = QPushButton("Start Record")
            self.record_button.clicked.connect(self.toggle_record)
            self.layout.addWidget(self.record_button)

            self.rotation_step = 5.0
            self.fov_step = 5.0
            self.res_step = 32

            self.connected = False
            self.last_check_time = time.time()

            self.toggle_buttons = {}

        self.central_widget.setLayout(self.layout)

        self.current_image = np.zeros((480, 640, 3), np.uint8)
        self.image_lock = threading.Lock()

        self.sub = session.declare_subscriber(view_keys[name], self.image_handler)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(10)

    def toggle_record(self):
        if self.is_recording:
            pub_record.put("stop".encode())
            self.record_button.setText("Start Record")
        else:
            pub_record.put("start".encode())
            self.record_button.setText("Stop Record")
        self.is_recording = not self.is_recording

    def image_handler(self, sample):
        try:
            buf = sample.payload.to_bytes()
            arr = np.frombuffer(buf, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                with self.image_lock:
                    self.current_image = img.copy()
        except Exception as e:
            print(f"Error decoding image for {self.name}: {e}")

    def update_image(self):
        with self.image_lock:
            height, width, _ = self.current_image.shape
            bytes_per_line = 3 * width
            qimg = QImage(self.current_image.tobytes(), width, height, bytes_per_line, QImage.Format.Format_BGR888)
            self.label.setPixmap(QPixmap.fromImage(qimg))

        self.label.adjustSize()
        self.adjustSize()

        if self.name == "forward":
            current_time = time.time()
            if current_time - self.last_check_time >= 1.0:
                peers = session.info.peers_zid()
                if not self.connected and len(peers) > 0:
                    print(f"Node 2 acknowledged connection to peer: {peers[0]}")
                    self.connected = True
                elif len(peers) == 0:
                    print("Node 2: No peers connected yet...")
                self.last_check_time = current_time

    def open_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Settings")
        layout = QFormLayout()

        fov_x_spin = QDoubleSpinBox()
        fov_x_spin.setRange(10.0, 180.0)
        fov_x_spin.setValue(self.shared_params.views[self.name]["fov_x"])
        layout.addRow("FOV X:", fov_x_spin)

        fov_y_spin = QDoubleSpinBox()
        fov_y_spin.setRange(10.0, 180.0)
        fov_y_spin.setValue(self.shared_params.views[self.name]["fov_y"])
        layout.addRow("FOV Y:", fov_y_spin)

        width_spin = QSpinBox()
        width_spin.setRange(100, 4000)
        width_spin.setValue(self.shared_params.views[self.name]["width"])
        layout.addRow("Width:", width_spin)

        height_spin = QSpinBox()
        height_spin.setRange(100, 4000)
        height_spin.setValue(self.shared_params.views[self.name]["height"])
        layout.addRow("Height:", height_spin)

        quality_spin = QSpinBox()
        quality_spin.setRange(0, 100)
        quality_spin.setValue(self.shared_params.views[self.name]["quality"])
        layout.addRow("Image Quality %:", quality_spin)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addRow(button_box)

        dialog.setLayout(layout)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.shared_params.views[self.name]["fov_x"] = fov_x_spin.value()
            self.shared_params.views[self.name]["fov_y"] = fov_y_spin.value()
            self.shared_params.views[self.name]["width"] = width_spin.value()
            self.shared_params.views[self.name]["height"] = height_spin.value()
            self.shared_params.views[self.name]["quality"] = quality_spin.value()
            self.shared_params.send(pub_params)
            self.shared_params.save_config()

    def keyPressEvent(self, event):
        if self.name != "forward":
            return super().keyPressEvent(event)

        key = event.key()
        self.pressed_keys.add(key)
        changed = False

        if key == Qt.Key.Key_Left:
            self.shared_params.yaw -= self.rotation_step
            changed = True
        elif key == Qt.Key.Key_Right:
            self.shared_params.yaw += self.rotation_step
            changed = True
        elif key == Qt.Key.Key_Up:
            self.shared_params.pitch += self.rotation_step
            changed = True
        elif key == Qt.Key.Key_Down:
            self.shared_params.pitch -= self.rotation_step
            changed = True
        elif key == Qt.Key.Key_A:
            self.shared_params.roll -= self.rotation_step
            changed = True
        elif key == Qt.Key.Key_D:
            self.shared_params.roll += self.rotation_step
            changed = True
        elif key == Qt.Key.Key_Plus:
            view = self.shared_params.views["forward"]
            view["fov_x"] += self.fov_step
            view["fov_y"] += self.fov_step
            changed = True
        elif key == Qt.Key.Key_Minus:
            view = self.shared_params.views["forward"]
            view["fov_x"] = max(10, view["fov_x"] - self.fov_step)
            view["fov_y"] = max(10, view["fov_y"] - self.fov_step)
            changed = True
        elif event.modifiers() & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_Right:
            self.shared_params.views["forward"]["width"] += self.res_step
            changed = True
        elif event.modifiers() & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_Left:
            self.shared_params.views["forward"]["width"] = max(320, self.shared_params.views["forward"]["width"] - self.res_step)
            changed = True
        elif event.modifiers() & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_Down:
            self.shared_params.views["forward"]["height"] += self.res_step
            changed = True
        elif event.modifiers() & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_Up:
            self.shared_params.views["forward"]["height"] = max(240, self.shared_params.views["forward"]["height"] - self.res_step)
            changed = True
        if Qt.Key.Key_Left in self.pressed_keys and Qt.Key.Key_Right in self.pressed_keys:
            self.shared_params.yaw = 0.0
            self.shared_params.pitch = 0.0
            self.shared_params.roll = 0.0
            view = self.shared_params.views["forward"]
            view["fov_x"] = 82.0
            view["fov_y"] = 82.0
            view["width"] = 640
            view["height"] = 480
            changed = True

        if changed:
            self.shared_params.send(pub_params)

    def keyReleaseEvent(self, event):
        if self.name != "forward":
            return super().keyReleaseEvent(event)
        key = event.key()
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)

    def add_toggle_button(self, view_name):
        button = QPushButton()
        self.toggle_buttons[view_name] = button
        self.update_button_text(view_name)
        button.clicked.connect(lambda: self.toggle_target(view_name))
        self.layout.addWidget(button)

    def update_button_text(self, view_name):
        button = self.toggle_buttons[view_name]
        state = "Close" if self.shared_params.views[view_name]["enable"] else "Open"
        button.setText(f"{state} {view_name.capitalize()}")

    def toggle_target(self, view_name):
        target = windows[view_name]
        if self.shared_params.views[view_name]["enable"]:
            target.hide()
            self.shared_params.views[view_name]["enable"] = False
        else:
            target.show()
            self.shared_params.views[view_name]["enable"] = True
        self.shared_params.send(pub_params)
        self.shared_params.save_config()
        self.update_button_text(view_name)

# Run PyQt6 app
app = QApplication([])

config = configparser.ConfigParser()
config.read("gui_config.cfg")

shared_params = SharedParams(config)

pub_params = session.declare_publisher("view_params")
pub_record = session.declare_publisher("record_command")

shared_params.send(pub_params)

windows = {}
for name in view_keys:
    windows[name] = ViewWindow(name, session, shared_params, view_keys)

# Arrange windows
windows["left"].move(100, 100)
windows["forward"].move(100 + 640 + 20, 100)
windows["right"].move(100 + 640 + 20 + 640 + 20, 100)
windows["rear"].move(100 + 640 + 20, 100 + 480 + 20)
windows["downward"].move(100 + 640 + 20 + 640 + 20, 100 + 480 + 20)

# Add toggle buttons to forward window
forward_win = windows["forward"]
forward_win.add_toggle_button("left")
forward_win.add_toggle_button("right")
forward_win.add_toggle_button("rear")
forward_win.add_toggle_button("downward")

# Show enabled windows
for name in view_keys:
    if shared_params.views[name]["enable"]:
        windows[name].show()

app.exec()

# Cleanup
session.close()