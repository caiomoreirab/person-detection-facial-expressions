"""
Detecção de múltiplos rostos e classificação de expressões em tempo real.

O programa usa:
- YOLOv11-Face para detectar TODOS os rostos do frame;
- AnyNet para classificar a expressão de cada rosto;
- ícones de emoji desenhados no próprio frame, sem depender de fontes externas.

Controles:
    Q  - encerra
    S  - salva o frame atual como PNG na pasta "capturas"
    +  - aumenta o limiar de confiança da YOLO
    -  - diminui o limiar de confiança da YOLO
    M  - alterna entre os modelos de classificação de emoções disponíveis
"""

import math
import urllib.request
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


# =============================================================================
# Arquitetura do classificador de expressões
# =============================================================================
class SEBlock(nn.Module):
    def __init__(self, ch, ratio=4):
        super().__init__()
        r = max(1, ch // ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(r, ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.se(x).view(x.size(0), -1, 1, 1)
        return x * w


class PlainConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=6, kernel_size=3, stride=1, **kwargs):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_ch,
                mid_ch,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=mid_ch,
                bias=False,
            ),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            SEBlock(mid_ch),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Stem(nn.Module):
    def __init__(self, out_ch=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.stem(x)


class AnyNetStage(nn.Module):
    def __init__(self, in_ch, out_ch, depth, block_type):
        super().__init__()
        blocks = []
        for i in range(depth):
            stride = 2 if i == 0 else 1
            blocks.append(block_type(in_ch if i == 0 else out_ch, out_ch, stride=stride))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class AnyNet(nn.Module):
    def __init__(self, num_stages, widths, depths, transition_stage, dropout, num_classes=8):
        super().__init__()
        self.stem = Stem(out_ch=32)
        stages = []
        in_ch = 32

        for i in range(num_stages):
            block_type = MBConv if i >= transition_stage - 1 else PlainConv
            stages.append(AnyNetStage(in_ch, widths[i], depths[i], block_type))
            in_ch = widths[i]

        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(widths[-1], num_classes),
        )

        last_stage = list(self.stages.children())[-1]
        last_block = list(last_stage.blocks.children())[-1]
        self.gradcam_target = last_block.conv[-1]

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)


def build_anynet(config, num_classes=8) -> AnyNet:
    return AnyNet(
        num_stages=config["num_stages"],
        widths=config["widths"],
        depths=config["depths"],
        transition_stage=config["transition_stage"],
        dropout=config["dropout"],
        num_classes=num_classes,
    )


# =============================================================================
# Configurações
# =============================================================================
# A detecção facial usa exclusivamente a YOLOv11n-Face.
YOLO_MODEL_NAME = "yolov11n-face.pt"

# Checkpoints de classificação de emoções procurados primeiro.
# Outros arquivos "anynet*.pth" presentes na pasta também serão encontrados.
PREFERRED_EMOTION_CHECKPOINTS = [
    "anynet_fold_multi_8.pth",
    "anynet_fold_1.pth",
    "anynet_fold_3.pth",
]
EMOTION_MODEL_IDX = 0
CONF_THRESH = 0.50
CONF_STEP = 0.05
CAMERA_INDEX = 0
WINDOW_NAME = "Deteccao de rostos e expressoes"
FACES_WINDOW_NAME = "Rostos classificados"

BASE_URL = "https://github.com/akanametov/yolo-face/"
BASE_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = BASE_DIR / "capturas"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cores BGR
COLOR_BOX = (0, 230, 118)
COLOR_LABEL_BG = (0, 230, 118)
COLOR_LABEL_TEXT = (15, 15, 15)
COLOR_HUD = (255, 0, 255)

# Nomes exibidos em português. A chave corresponde ao nome salvo no checkpoint.
EMOTION_LABELS = {
    "angry": "Raiva",
    "contempt": "Desprezo",
    "disgust": "Nojo",
    "fear": "Medo",
    "happy": "Feliz",
    "neutral": "Neutro",
    "sad": "Triste",
    "surprise": "Surpresa",
    "suprise": "Surpresa",  # grafia presente no checkpoint original
}

TRANSFORM_EVAL = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
)


# =============================================================================
# Carregamento dos modelos
# =============================================================================
def ensure_weights(model_name: str) -> str:
    """Retorna o caminho do peso e tenta baixá-lo caso esteja ausente."""
    path = BASE_DIR / model_name
    if not path.exists():
        url = BASE_URL + model_name
        print(f"[INFO] Baixando {model_name} de {url} ...")
        urllib.request.urlretrieve(url, path)
        print(f"[INFO] {model_name} salvo em {path.resolve()}")
    return str(path)


def load_yolo_model(model_name: str) -> YOLO:
    weights = ensure_weights(model_name)
    print(f"[INFO] Carregando {model_name} em {DEVICE}...")
    return YOLO(weights)


def discover_emotion_checkpoints():
    """
    Localiza os checkpoints de classificação de emoções na pasta do programa.

    A ordem começa pelos nomes preferenciais e depois inclui outros arquivos
    cujo nome siga o padrão "anynet*.pth".
    """
    discovered = []
    seen = set()

    for checkpoint_name in PREFERRED_EMOTION_CHECKPOINTS:
        checkpoint_path = BASE_DIR / checkpoint_name
        if checkpoint_path.exists():
            resolved = checkpoint_path.resolve()
            if resolved not in seen:
                discovered.append(checkpoint_path)
                seen.add(resolved)

    for checkpoint_path in sorted(BASE_DIR.glob("anynet*.pth")):
        resolved = checkpoint_path.resolve()
        if resolved not in seen:
            discovered.append(checkpoint_path)
            seen.add(resolved)

    if not discovered:
        expected = ", ".join(PREFERRED_EMOTION_CHECKPOINTS)
        raise FileNotFoundError(
            "Nenhum checkpoint de classificação de emoções foi encontrado em "
            f"{BASE_DIR}. Arquivos esperados: {expected}"
        )

    return discovered


def get_emotion_model_display_name(checkpoint_path: Path, checkpoint: dict) -> str:
    """Gera o nome mostrado no HUD para o classificador em uso."""
    metadata_name = (
        checkpoint.get("model_name")
        or checkpoint.get("architecture")
        or checkpoint.get("arch")
    )

    if metadata_name:
        return f"{metadata_name} ({checkpoint_path.stem})"

    return f"AnyNet ({checkpoint_path.stem})"


def load_emotion_model(checkpoint_path: Path):
    """Carrega um checkpoint específico do classificador de emoções."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")

    # weights_only=False é necessário porque o checkpoint contém configuração e classes.
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        # Compatibilidade com versões mais antigas do PyTorch.
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    required_keys = {"classes", "config", "model_state_dict"}
    missing_keys = required_keys.difference(checkpoint.keys())
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise KeyError(
            f"O checkpoint {checkpoint_path.name} não possui as chaves necessárias: {missing}"
        )

    class_names = checkpoint["classes"]
    emotion_model = build_anynet(
        checkpoint["config"],
        num_classes=len(class_names),
    ).to(DEVICE)
    emotion_model.load_state_dict(checkpoint["model_state_dict"])
    emotion_model.eval()

    display_name = get_emotion_model_display_name(checkpoint_path, checkpoint)

    print(
        f"[INFO] Classificador carregado: {display_name}. "
        f"Classes: {', '.join(class_names)}"
    )
    return emotion_model, class_names, display_name


# =============================================================================
# Inferência
# =============================================================================
def predict_faces(model: YOLO, frame, conf_thresh: float):
    results = model.predict(
        source=frame,
        conf=conf_thresh,
        device=DEVICE,
        verbose=False,
    )
    return results[0]


def predict_emotion(emotion_model, face_bgr, class_names):
    if face_bgr is None or face_bgr.size == 0:
        raise ValueError("O recorte do rosto está vazio.")

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    tensor = TRANSFORM_EVAL(face_pil).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = emotion_model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred_index = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_index].item())

    return class_names[pred_index], confidence


# =============================================================================
# Emojis desenhados no frame
# =============================================================================
def _normalize_emotion(emotion: str) -> str:
    emotion = emotion.lower().strip()
    return "surprise" if emotion == "suprise" else emotion


@lru_cache(maxsize=32)
def make_emoji_icon(emotion: str, size: int = 36):
    """Cria um pequeno emoji BGRA. Não depende de fonte ou arquivo PNG externo."""
    emotion = _normalize_emotion(emotion)
    icon = np.zeros((size, size, 4), dtype=np.uint8)

    cx = size // 2
    cy = size // 2
    radius = max(3, int(size * 0.43))
    thickness = max(1, size // 18)

    black = (20, 20, 20, 255)
    white = (255, 255, 255, 255)
    yellow = (0, 220, 255, 255)
    green = (80, 200, 100, 255)
    blue = (255, 190, 90, 255)

    face_color = green if emotion == "disgust" else blue if emotion == "fear" else yellow
    cv2.circle(icon, (cx, cy), radius, face_color, -1, cv2.LINE_AA)
    cv2.circle(icon, (cx, cy), radius, black, thickness, cv2.LINE_AA)

    left_eye = (int(size * 0.35), int(size * 0.40))
    right_eye = (int(size * 0.65), int(size * 0.40))
    eye_r = max(1, size // 18)

    if emotion == "happy":
        # Olhos sorrindo e boca aberta.
        cv2.ellipse(icon, left_eye, (eye_r + 2, eye_r + 1), 0, 200, 340, black, thickness, cv2.LINE_AA)
        cv2.ellipse(icon, right_eye, (eye_r + 2, eye_r + 1), 0, 200, 340, black, thickness, cv2.LINE_AA)
        cv2.ellipse(
            icon,
            (cx, int(size * 0.59)),
            (int(size * 0.19), int(size * 0.14)),
            0,
            0,
            180,
            black,
            max(2, thickness + 1),
            cv2.LINE_AA,
        )

    elif emotion == "sad":
        cv2.circle(icon, left_eye, eye_r, black, -1, cv2.LINE_AA)
        cv2.circle(icon, right_eye, eye_r, black, -1, cv2.LINE_AA)
        cv2.ellipse(
            icon,
            (cx, int(size * 0.73)),
            (int(size * 0.18), int(size * 0.14)),
            0,
            190,
            350,
            black,
            thickness,
            cv2.LINE_AA,
        )
        # Lágrima.
        cv2.ellipse(
            icon,
            (int(size * 0.70), int(size * 0.55)),
            (max(1, size // 24), max(2, size // 12)),
            0,
            0,
            360,
            (255, 120, 0, 255),
            -1,
            cv2.LINE_AA,
        )

    elif emotion == "angry":
        cv2.circle(icon, left_eye, eye_r, black, -1, cv2.LINE_AA)
        cv2.circle(icon, right_eye, eye_r, black, -1, cv2.LINE_AA)
        cv2.line(
            icon,
            (int(size * 0.25), int(size * 0.28)),
            (int(size * 0.44), int(size * 0.36)),
            black,
            thickness,
            cv2.LINE_AA,
        )
        cv2.line(
            icon,
            (int(size * 0.75), int(size * 0.28)),
            (int(size * 0.56), int(size * 0.36)),
            black,
            thickness,
            cv2.LINE_AA,
        )
        cv2.ellipse(
            icon,
            (cx, int(size * 0.72)),
            (int(size * 0.18), int(size * 0.12)),
            0,
            190,
            350,
            black,
            thickness,
            cv2.LINE_AA,
        )

    elif emotion == "fear":
        cv2.circle(icon, left_eye, eye_r + 1, white, -1, cv2.LINE_AA)
        cv2.circle(icon, right_eye, eye_r + 1, white, -1, cv2.LINE_AA)
        cv2.circle(icon, left_eye, max(1, eye_r // 2), black, -1, cv2.LINE_AA)
        cv2.circle(icon, right_eye, max(1, eye_r // 2), black, -1, cv2.LINE_AA)
        cv2.ellipse(
            icon,
            (cx, int(size * 0.66)),
            (int(size * 0.10), int(size * 0.15)),
            0,
            0,
            360,
            black,
            -1,
            cv2.LINE_AA,
        )

    elif emotion == "surprise":
        cv2.circle(icon, left_eye, eye_r + 1, white, -1, cv2.LINE_AA)
        cv2.circle(icon, right_eye, eye_r + 1, white, -1, cv2.LINE_AA)
        cv2.circle(icon, left_eye, max(1, eye_r // 2), black, -1, cv2.LINE_AA)
        cv2.circle(icon, right_eye, max(1, eye_r // 2), black, -1, cv2.LINE_AA)
        cv2.circle(icon, (cx, int(size * 0.68)), max(2, size // 10), black, thickness, cv2.LINE_AA)

    elif emotion == "disgust":
        cv2.line(
            icon,
            (int(size * 0.27), int(size * 0.38)),
            (int(size * 0.42), int(size * 0.42)),
            black,
            thickness,
            cv2.LINE_AA,
        )
        cv2.line(
            icon,
            (int(size * 0.58), int(size * 0.42)),
            (int(size * 0.73), int(size * 0.38)),
            black,
            thickness,
            cv2.LINE_AA,
        )
        points = np.array(
            [
                [int(size * 0.33), int(size * 0.67)],
                [int(size * 0.44), int(size * 0.62)],
                [int(size * 0.55), int(size * 0.69)],
                [int(size * 0.68), int(size * 0.64)],
            ],
            dtype=np.int32,
        )
        cv2.polylines(icon, [points], False, black, thickness, cv2.LINE_AA)

    elif emotion == "contempt":
        cv2.circle(icon, left_eye, eye_r, black, -1, cv2.LINE_AA)
        cv2.line(
            icon,
            (int(size * 0.58), int(size * 0.40)),
            (int(size * 0.72), int(size * 0.36)),
            black,
            thickness,
            cv2.LINE_AA,
        )
        cv2.ellipse(
            icon,
            (int(size * 0.56), int(size * 0.61)),
            (int(size * 0.20), int(size * 0.10)),
            0,
            10,
            150,
            black,
            thickness,
            cv2.LINE_AA,
        )

    else:  # neutral
        cv2.circle(icon, left_eye, eye_r, black, -1, cv2.LINE_AA)
        cv2.circle(icon, right_eye, eye_r, black, -1, cv2.LINE_AA)
        cv2.line(
            icon,
            (int(size * 0.36), int(size * 0.68)),
            (int(size * 0.64), int(size * 0.68)),
            black,
            thickness,
            cv2.LINE_AA,
        )

    return icon


def overlay_bgra(background, foreground, x: int, y: int):
    """Sobrepõe uma imagem BGRA sobre um frame BGR, respeitando bordas e alpha."""
    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = foreground.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + fg_w)
    y2 = min(bg_h, y + fg_h)

    if x1 >= x2 or y1 >= y2:
        return

    fg_x1 = x1 - x
    fg_y1 = y1 - y
    fg_x2 = fg_x1 + (x2 - x1)
    fg_y2 = fg_y1 + (y2 - y1)

    fg_crop = foreground[fg_y1:fg_y2, fg_x1:fg_x2]
    alpha = fg_crop[:, :, 3:4].astype(np.float32) / 255.0
    bg_crop = background[y1:y2, x1:x2].astype(np.float32)

    blended = alpha * fg_crop[:, :, :3].astype(np.float32) + (1.0 - alpha) * bg_crop
    background[y1:y2, x1:x2] = blended.astype(np.uint8)


# =============================================================================
# Desenho das detecções
# =============================================================================
def draw_emotion_label(frame, x1, y1, y2, emotion, emotion_confidence):
    frame_h, frame_w = frame.shape[:2]
    emoji_size = 34
    padding = 5
    gap = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    thickness = 2

    display_name = EMOTION_LABELS.get(_normalize_emotion(emotion), emotion.capitalize())
    text = f"{display_name} {emotion_confidence * 100:.0f}%"
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    label_w = padding + emoji_size + gap + text_w + padding
    label_h = max(emoji_size, text_h + baseline) + 2 * padding

    label_x = min(max(0, x1), max(0, frame_w - label_w))

    # Primeiro tenta colocar acima da caixa. Se isso invadir a área do HUD
    # (canto superior esquerdo), coloca abaixo do rosto.
    above_y = y1 - label_h
    below_y = y2 + 2
    overlaps_hud = label_x < 400 and above_y < 125

    if above_y >= 0 and not overlaps_hud:
        label_y = above_y
    elif below_y + label_h <= frame_h:
        label_y = below_y
    else:
        # Último recurso: coloca dentro da caixa, próximo ao topo.
        label_y = min(max(0, y1 + 2), max(0, frame_h - label_h))

    cv2.rectangle(
        frame,
        (label_x, label_y),
        (label_x + label_w, label_y + label_h),
        COLOR_LABEL_BG,
        -1,
        cv2.LINE_AA,
    )

    emoji = make_emoji_icon(emotion, emoji_size)
    emoji_y = label_y + (label_h - emoji_size) // 2
    overlay_bgra(frame, emoji, label_x + padding, emoji_y)

    text_x = label_x + padding + emoji_size + gap
    text_y = label_y + (label_h + text_h - baseline) // 2
    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        font,
        font_scale,
        COLOR_LABEL_TEXT,
        thickness,
        cv2.LINE_AA,
    )


def draw_hud(frame, emotion_model_name: str, conf_thresh: float, n_faces: int):
    hud = [
        f"Modelo: {emotion_model_name}",
        f"Device: {DEVICE}",
        f"Conf. YOLO: {conf_thresh:.2f} (+/-)",
        f"Rostos detectados: {n_faces}",
        "[Q] Sair  [S] Salvar  [M] Trocar modelo",
    ]

    # Fundo escuro para manter o HUD legível.
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (390, 118), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(hud):
        cv2.putText(
            frame,
            line,
            (12, 24 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            COLOR_HUD,
            1,
            cv2.LINE_AA,
        )


def build_faces_grid(face_entries, tile_size=112, columns=4):
    """Monta uma grade com todos os rostos classificados no frame atual."""
    label_height = 30

    if not face_entries:
        blank = np.zeros((tile_size + label_height, tile_size * 2, 3), dtype=np.uint8)
        cv2.putText(
            blank,
            "Nenhum rosto detectado",
            (10, tile_size // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        return blank

    columns = max(1, min(columns, len(face_entries)))
    rows = math.ceil(len(face_entries) / columns)
    grid = np.zeros((rows * (tile_size + label_height), columns * tile_size, 3), dtype=np.uint8)

    for index, entry in enumerate(face_entries):
        row = index // columns
        col = index % columns
        x = col * tile_size
        y = row * (tile_size + label_height)

        resized = cv2.resize(entry["face"], (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        grid[y : y + tile_size, x : x + tile_size] = resized

        display_name = EMOTION_LABELS.get(
            _normalize_emotion(entry["emotion"]),
            entry["emotion"].capitalize(),
        )
        short_label = f"{display_name} {entry['emotion_confidence'] * 100:.0f}%"
        cv2.putText(
            grid,
            short_label,
            (x + 4, y + tile_size + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        emoji = make_emoji_icon(entry["emotion"], 24)
        overlay_bgra(grid, emoji, x + tile_size - 28, y + tile_size + 2)

    return grid


def process_all_faces(frame, result, emotion_model, class_names):
    """Classifica e desenha todas as faces detectadas pela YOLO."""
    frame_h, frame_w = frame.shape[:2]
    face_entries = []

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return face_entries

    # Ordena as caixas da esquerda para a direita para estabilizar a grade.
    boxes_list = list(boxes)
    boxes_list.sort(key=lambda box: float(box.xyxy[0][0]))

    for box in boxes_list:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detection_confidence = float(box.conf[0].item())

        x1 = max(0, min(x1, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        x2 = max(0, min(x2, frame_w))
        y2 = max(0, min(y2, frame_h))

        if x2 <= x1 or y2 <= y1:
            continue

        face = frame[y1:y2, x1:x2].copy()
        if face.size == 0:
            continue

        try:
            emotion, emotion_confidence = predict_emotion(
                emotion_model,
                face,
                class_names,
            )
        except Exception as exc:
            print(f"[WARN] Não foi possível classificar um rosto: {exc}")
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2, cv2.LINE_AA)
        draw_emotion_label(frame, x1, y1, y2, emotion, emotion_confidence)

        face_entries.append(
            {
                "face": face,
                "emotion": emotion,
                "emotion_confidence": emotion_confidence,
                "detection_confidence": detection_confidence,
                "box": (x1, y1, x2, y2),
            }
        )

    return face_entries


# =============================================================================
# Loop principal
# =============================================================================
def main():
    global EMOTION_MODEL_IDX

    # A YOLO permanece fixa durante toda a execução.
    yolo_model = load_yolo_model(YOLO_MODEL_NAME)

    emotion_checkpoints = discover_emotion_checkpoints()
    EMOTION_MODEL_IDX = min(EMOTION_MODEL_IDX, len(emotion_checkpoints) - 1)

    emotion_model, class_names, emotion_model_name = load_emotion_model(
        emotion_checkpoints[EMOTION_MODEL_IDX]
    )

    print("[INFO] Modelos de emoção disponíveis:")
    for index, checkpoint_path in enumerate(emotion_checkpoints, start=1):
        marker = " (ativo)" if index - 1 == EMOTION_MODEL_IDX else ""
        print(f"       {index}. {checkpoint_path.name}{marker}")

    conf = CONF_THRESH
    saved = 0
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Câmera {CAMERA_INDEX} não encontrada. "
            "Caso tenha mais de uma câmera, altere CAMERA_INDEX para 1."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] Câmera aberta. Q=sair | S=salvar | M=trocar classificador | +/-=confiança YOLO")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame não capturado.")
                break

            result = predict_faces(yolo_model, frame, conf)
            face_entries = process_all_faces(frame, result, emotion_model, class_names)
            draw_hud(frame, emotion_model_name, conf, len(face_entries))

            faces_grid = build_faces_grid(face_entries)
            cv2.imshow(WINDOW_NAME, frame)
            cv2.imshow(FACES_WINDOW_NAME, faces_grid)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

            if key in (ord("s"), ord("S")):
                filename = CAPTURE_DIR / f"face_frame_{saved:04d}.png"
                cv2.imwrite(str(filename), frame)
                saved += 1
                print(f"[INFO] Salvo: {filename}")

            elif key in (ord("+"), ord("=")):
                conf = min(conf + CONF_STEP, 0.95)
                print(f"[INFO] Confiança YOLO -> {conf:.2f}")

            elif key == ord("-"):
                conf = max(conf - CONF_STEP, 0.05)
                print(f"[INFO] Confiança YOLO -> {conf:.2f}")

            elif key in (ord("m"), ord("M")):
                if len(emotion_checkpoints) == 1:
                    print(
                        "[INFO] Existe apenas um modelo de classificação de emoções "
                        f"disponível: {emotion_checkpoints[0].name}"
                    )
                    continue

                previous_idx = EMOTION_MODEL_IDX
                previous_model = emotion_model
                loaded = False

                # Tenta os próximos checkpoints até encontrar um compatível.
                for step in range(1, len(emotion_checkpoints)):
                    candidate_idx = (previous_idx + step) % len(emotion_checkpoints)
                    candidate_path = emotion_checkpoints[candidate_idx]

                    try:
                        new_model, new_classes, new_display_name = load_emotion_model(
                            candidate_path
                        )
                    except Exception as exc:
                        print(
                            f"[WARN] Não foi possível carregar {candidate_path.name}: {exc}"
                        )
                        continue

                    EMOTION_MODEL_IDX = candidate_idx
                    emotion_model = new_model
                    class_names = new_classes
                    emotion_model_name = new_display_name
                    loaded = True

                    del previous_model
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()

                    print(
                        "[INFO] Modelo de classificação de emoções trocado para "
                        f"{emotion_model_name}"
                    )
                    break

                if not loaded:
                    print(
                        "[WARN] Nenhum outro checkpoint compatível pôde ser carregado. "
                        "O modelo atual foi mantido."
                    )

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Encerrado.")


if __name__ == "__main__":
    main()