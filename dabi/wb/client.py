import websocket
import threading
import time
import json
import traceback

class SyncWebSocketClient:
    def __init__(self, url, heartbeat_interval=10):
        self.url = url
        self.heartbeat_interval = heartbeat_interval
        self.ws = None
        self.connected = False
        self.running = False
        self.lock = threading.Lock()
        self.heartbeat_thread = None
        self.receiver_thread = None

    def connect(self):
        try:
            print(f"🚀 Connecting to {self.url}")
            self.ws = websocket.WebSocket()
            self.ws.connect(self.url)
            self.connected = True
            print("✅ Connected.")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            traceback.print_exc()
            self.connected = False

    def send_message(self, message):
        with self.lock:
            if not self.connected:
                print("⚠️ Cannot send: not connected")
                self.reconnect()
                return
            try:
                if isinstance(message, dict):
                    message = json.dumps(message)
                self.ws.send(message)
                print(f"📤 Sent: {message}")
            except Exception as e:
                print(f"❌ Send failed: {e}")
                self.connected = False

    def receive_messages(self):
        while self.running:
            if not self.connected:
                self.reconnect()
                time.sleep(1)
                continue
            try:
                message = self.ws.recv()
                if message:
                    print(f"📩 Received: {message}")
            except Exception as e:
                print(f"⚠️ Receive failed: {e}")
                self.connected = False
                time.sleep(1)

    def send_heartbeat(self):
        while self.running:
            time.sleep(self.heartbeat_interval)
            if not self.connected:
                self.reconnect()
                continue
            try:
                self.ws.send("ping")
                print("🔁 Sent heartbeat ping")
            except Exception as e:
                print(f"❤️‍🔥 Heartbeat failed: {e}")
                self.connected = False
                self.reconnect()

    def reconnect(self):
        print("🔄 Reconnecting...")
        try:
            self.ws.close()
        except:
            pass
        self.connect()

    def start(self):
        self.running = True
        self.connect()

        # 启动心跳检测线程
        self.heartbeat_thread = threading.Thread(target=self.send_heartbeat, daemon=True)
        self.heartbeat_thread.start()
        self.receiver_thread = threading.Thread(target=self.receive_messages, daemon=True)
        self.receiver_thread.start()

    def stop(self):
        print("🛑 Stopping client...")
        self.running = False
        try:
            self.ws.close()
        except:
            pass
        self.connected = False
        print("🔒 Client stopped.")

