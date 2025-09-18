# receiver_macos.py
import socket
import subprocess
import shlex

HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 5001       # Must match sender's port

def speak(text):
    if text.strip():
        print(f"[MacOS] Speaking text: {text}")
        cmd = f'say -v Alex "{text}"'
        subprocess.run(shlex.split(cmd))
    else:
        print("[MacOS] Received empty text, skipping speech.")

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[MacOS] Listening on port {PORT}...")

        while True:  # Keep listening for multiple messages
            conn, addr = s.accept()
            with conn:
                print(f"[MacOS] Connection from {addr}")
                data = conn.recv(4096).decode("utf-8")
                if data:
                    print(f"[MacOS] Received text:\n{data}")
                    speak(data)

if __name__ == "__main__":
    main()
