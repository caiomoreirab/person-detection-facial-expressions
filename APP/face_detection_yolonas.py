"""
Detecção de rosto em tempo real via webcam com YOLOv11-Face
============================================================
Repositório dos pesos: https://github.com/akanametov/yolo-face

Instalação:
    pip install ultralytics opencv-python

Na primeira execução o script baixa o peso automaticamente (~6 MB para a
variante 'n'). Versões disponíveis: yolov11n / s / m / l -face.pt

Controles:
    Q  — encerra
    S  — salva o frame atual como PNG
    +  — aumenta limiar de confiança
    -  — diminui limiar de confiança
    M  — alterna entre modelos (n → s → m → l)
"""

import cv2
import torch
import urllib.request
from pathlib import Path
from ultralytics import YOLO

# ── Configurações ──────────────────────────────────────────────────────────────
MODELS = [
    "yolov11n-face.pt",   # nano  — mais rápido
    "yolov11s-face.pt",   # small
    "yolov11m-face.pt",   # medium
    "yolov11l-face.pt",   # large — mais preciso
]
MODEL_IDX    = 0          # começa com o nano
CONF_THRESH  = 0.50
CONF_STEP    = 0.05
CAMERA_INDEX = 0
WINDOW_NAME  = "YOLOv11-Face Detection"

BASE_URL = "https://github.com/akanametov/yolo-face/"

# Cores (BGR)
COLOR_BOX   = (0, 230, 118)
COLOR_HUD   = (255, 0, 255)
COLOR_LABEL = (0, 0, 0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).resolve().parent


# ── Download automático do peso se não existir ─────────────────────────────────
def ensure_weights(model_name: str) -> str:
    path = BASE_DIR / model_name
    if not path.exists():
        url = BASE_URL + model_name
        print(f"[INFO] Baixando {model_name} de {url} ...")
        urllib.request.urlretrieve(url, path)
        print(f"[INFO] {model_name} salvo em {path.resolve()}")
    return str(path)


# ── Carrega modelo ─────────────────────────────────────────────────────────────
def load_model(model_name: str) -> YOLO:
    weights = ensure_weights(model_name)
    print(f"[INFO] Carregando {model_name} em {DEVICE}...")
    model = YOLO(weights)
    return model


# ── Inferência num frame ───────────────────────────────────────────────────────
def predict(model: YOLO, frame, conf_thresh: float):
    results = model.predict(
        source=frame,
        conf=conf_thresh,
        device=DEVICE,
        verbose=False,
    )
    return results[0]   # ImageDetectionPrediction para o frame


# ── Desenha detecções + HUD ────────────────────────────────────────────────────
def draw(frame, result, conf_thresh: float, model_name: str):
    boxes = result.boxes

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)

        # Label de confiança
        label = f"Face {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame,
                      (x1, y1 - th - 8),
                      (x1 + tw + 6, y1),
                      COLOR_BOX, -1)
        cv2.putText(frame, label,
                    (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    COLOR_LABEL, 1, cv2.LINE_AA)

    # HUD
    n_faces = len(boxes)
    hud = [
        f"Modelo  : {model_name}",
        f"Device  : {DEVICE}",
        f"Conf    : {conf_thresh:.2f}  (+/-)",
        f"Rostos  : {n_faces}",
        "[Q] Sair  [S] Salvar  [M] Trocar modelo",
    ]
    for i, line in enumerate(hud):
        cv2.putText(frame, line, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    COLOR_HUD, 1, cv2.LINE_AA)

    return frame


# ── Loop principal ─────────────────────────────────────────────────────────────
def main():
    global MODEL_IDX

    model_name = MODELS[MODEL_IDX]
    model      = load_model(model_name)
    conf       = CONF_THRESH
    saved      = 0

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Câmera {CAMERA_INDEX} não encontrada.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] Câmera aberta. Q=sair | S=salvar | M=trocar modelo | +/-=confiança")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame não capturado.")
            break

        result = predict(model, frame, conf)
        frame  = draw(frame, result, conf, model_name)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

        elif key in (ord("s"), ord("S")):
            fname = f"face_frame_{saved:04d}.png"
            cv2.imwrite(fname, frame)
            saved += 1
            print(f"[INFO] Salvo: {fname}")

        elif key in (ord("+"), ord("=")):
            conf = min(conf + CONF_STEP, 0.95)
            print(f"[INFO] Confiança → {conf:.2f}")

        elif key == ord("-"):
            conf = max(conf - CONF_STEP, 0.05)
            print(f"[INFO] Confiança → {conf:.2f}")

        elif key in (ord("m"), ord("M")):
            MODEL_IDX  = (MODEL_IDX + 1) % len(MODELS)
            model_name = MODELS[MODEL_IDX]
            model      = load_model(model_name)
            print(f"[INFO] Modelo trocado para {model_name}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Encerrado.")


if __name__ == "__main__":
    main()
