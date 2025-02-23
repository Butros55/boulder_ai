import subprocess

def start_backend():
    try:
        subprocess.run(["docker-compose", "up", "--build"], check=False)
    except subprocess.CalledProcessError as e:
        print("Fehler beim Starten des Backends:", e)

if __name__ == "__main__":
    start_backend()
