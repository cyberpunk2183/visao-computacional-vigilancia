#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agente local de vigilancia por RTSP.

Alarme atual:
  USO_CELULAR_FUNCIONARIO_15S
  CONSUMO_MERCADORIA_LOJA

Regra:
  - monitora 32 cameras RTSP
  - analisa 1 camera por vez durante 50 segundos
  - dispara quando ve funcionario provavel usando celular por 15 segundos
    acumulados dentro da janela de 50 segundos
  - nos canais de hortifruti/laticinios, dispara quando ve pessoa com mao
    perto da boca por tempo acumulado, indicando possivel consumo de mercadoria
"""

import cv2
import numpy as np
import os
import json
import re
import signal
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread


# ---------------------------------------------------------------------------
# Configuracoes
# ---------------------------------------------------------------------------
# As câmeras são configuradas por variáveis de ambiente, para que nenhuma
# credencial ou IP de DVR fique no código. Veja .env.example.
#   RTSP_BASE_LOJA1="rtsp://usuario:senha@HOST:554/cam/realmonitor?channel={}&subtype=1"
#   CANAIS_LOJA1=32
# O placeholder {} é substituído pelo número do canal.
import os as _os
def _loja(nome, env_base, env_canais, canais_padrao):
    base = _os.environ.get(env_base,
        "rtsp://usuario:senha@192.0.2.10:554/cam/realmonitor?channel={}&subtype=1")
    return {"nome": nome, "rtsp_base": base,
            "canais": int(_os.environ.get(env_canais, canais_padrao))}

LOJAS_CAMERAS = [
    _loja("loja 1", "RTSP_BASE_LOJA1", "CANAIS_LOJA1", 32),
    _loja("loja 2", "RTSP_BASE_LOJA2", "CANAIS_LOJA2", 18),
]
NUM_CAMERAS = sum(loja["canais"] for loja in LOJAS_CAMERAS)
CAMERAS_GRUPO = 1
SEGUNDOS_GRUPO = 50

FRAME_RATE = 10
CONF_PESSOA = 0.20
CONF_CELULAR = 0.10
YOLO_IMGSZ = 640
YOLO_CELULAR_CROP_IMGSZ = 480
POSE_IMGSZ = 320
CLAHE_CLIPL = 2.5      # pre-processamento de contraste antes do YOLO
USAR_POSE_MAO_ROSTO = True
USAR_CROP_CELULAR = False
IA_INTERVALO_SEG = 1.0

TEMPO_CELULAR_ALERTA = 15
TEMPO_CONSUMO_ALERTA = 12
SCORE_MIN_CONSUMO_MAO_BOCA = 0.42
LIMITE_BOCA_PROPORCAO = 0.22
DECAIMENTO_CELULAR_SEM_EVIDENCIA = 2.0
DECAIMENTO_CONSUMO_SEM_EVIDENCIA = 3.0
MIN_CONF_KEYPOINT_ROSTO = 0.40
LIMIAR_MOVIMENTO_PROPORCAO = 0.004
COOLDOWN_ALERTA_SEG = 600
BUF_VIDEO_SEG = 45
TIMEOUT_RTSP_MS = 5000
SEM_FRAME_REINICIAR_SEG = 12
IA_TRAVADA_REINICIAR_SEG = 25
PAUSA_REINICIO_CAMERA_SEG = 3
REINICIAR_PROCESSO_EM_ERRO = True
PAUSA_REINICIO_PROCESSO_SEG = 5
DEBUG_INTERVALO_SEG = 5
DEBUG_DETECCOES = False
MOSTRAR_VIDEO = True

AZUL_UNIFORME_HSV_MIN = np.array([90, 45, 35])
AZUL_UNIFORME_HSV_MAX = np.array([135, 255, 255])
PROPORCAO_MIN_AZUL_TRONCO = 0.18
BRANCO_UNIFORME_HSV_MIN = np.array([0, 0, 175])
BRANCO_UNIFORME_HSV_MAX = np.array([179, 45, 255])
PROPORCAO_MIN_BRANCO_TRONCO = 0.45

# Preencha quando souber quais cameras sao de areas restritas de funcionarios.
# Exemplo: CAMERAS_RESTRITAS_BRANCO = {8, 9, 10}
CAMERAS_RESTRITAS_BRANCO = {23}

# Opcional: restrinja por poligonos dentro de cada camera.
# Se o canal estiver aqui, branco so conta dentro do poligono.
# Exemplo:
# ZONAS_RESTRITAS_BRANCO = {
#     8: [[(120, 80), (600, 80), (620, 340), (100, 340)]],
# }
ZONAS_RESTRITAS_BRANCO = {}

TIPO_ALERTA_CELULAR = "USO_CELULAR_FUNCIONARIO_15S"
TIPO_ALERTA_CONSUMO = "CONSUMO_MERCADORIA_LOJA"

# Canais onde faz sentido procurar possivel consumo de mercadoria.
# Valores das zonas: x1, y1, x2, y2 em proporcao do frame.
AREAS_CONSUMO_MERCADORIA = {
    4: {"nome": "hortifruti", "zona": (0.0, 0.0, 1.0, 1.0)},
    19: {"nome": "laticinios_direita", "zona": (0.50, 0.0, 1.0, 1.0)},
}

BASE_DIR = Path("vigilancia")
LOGS_DIR = BASE_DIR / "logs"
FRAMES_DIR = BASE_DIR / "frames"
VIDEOS_DIR = BASE_DIR / "videos"
DEBUG_DIR = BASE_DIR / "debug"
STATE_PATH = BASE_DIR / "estado_monitoramento.json"
ULTRALYTICS_DIR = (BASE_DIR / "ultralytics").resolve()
MPLCONFIG_DIR = (BASE_DIR / "matplotlib").resolve()

ULTRALYTICS_DIR.mkdir(parents=True, exist_ok=True)
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))


# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------
YOLO_DISPONIVEL = False
modelo_yolo = None
modelo_pose = None
yolo_lock = Lock()
STOP_EVENT = Event()
PULAR_DIRECAO = [0]   # 0=normal, 1=proxima, -1=anterior
CAMERA_TRAVADA = [False]  # True = fica na camera atual indefinidamente


def solicitar_parada(signum=None, frame=None):
    STOP_EVENT.set()
    print("\n[STOP] Parada solicitada. Encerrando recursos...")


signal.signal(signal.SIGINT, solicitar_parada)


class CameraTravada(RuntimeError):
    pass


def tem_movimento(frame, frame_anterior):
    if frame_anterior is None:
        return True
    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    small_ant = cv2.resize(frame_anterior, (0, 0), fx=0.25, fy=0.25)
    diff = cv2.absdiff(
        cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(small_ant, cv2.COLOR_BGR2GRAY),
    )
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    return cv2.countNonZero(thresh) / thresh.size >= LIMIAR_MOVIMENTO_PROPORCAO

try:
    from ultralytics import YOLO

    print("[INFO] Carregando YOLOv8 Nano...")
    modelo_yolo = YOLO("yolov8n.pt")
    if USAR_POSE_MAO_ROSTO:
        print("[INFO] Carregando YOLOv8 Pose Nano...")
        modelo_pose = YOLO("yolov8n-pose.pt")
    YOLO_DISPONIVEL = True
    print("[INFO] YOLOv8 Nano carregado.")
except ImportError:
    print("[ERRO] ultralytics nao instalado. Rode instalar.bat antes de iniciar.")
except Exception as e:
    print(f"[ERRO] Falha ao carregar YOLO: {e}")


# ---------------------------------------------------------------------------
# Setup e logs
# ---------------------------------------------------------------------------
def setup():
    for diretorio in [LOGS_DIR, FRAMES_DIR, VIDEOS_DIR, DEBUG_DIR]:
        diretorio.mkdir(parents=True, exist_ok=True)

    hoje = datetime.now().strftime("%Y%m%d")
    return LOGS_DIR / f"incidentes_{hoje}.txt"


def escrever_log(log_path, mensagem):
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    linha = f"[{timestamp}] {mensagem}\n"
    print(linha, end="")
    with open(log_path, "a", encoding="utf-8") as arquivo:
        arquivo.write(linha)


def escrever_incidente(log_path, numero, canal, descricao, frame_path, video_path, tipo=TIPO_ALERTA_CELULAR):
    bloco = (
        f"\n{'=' * 60}\n"
        f"  INCIDENTE #{numero}\n"
        f"  Tipo      : {tipo}\n"
        f"  Timestamp : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
        f"  Camera    : Canal {canal:02d}\n"
        f"  Descricao : {descricao}\n"
        f"  Frame     : {frame_path}\n"
        f"  Video     : {video_path}\n"
        f"{'=' * 60}\n"
    )
    print(bloco)
    with open(log_path, "a", encoding="utf-8") as arquivo:
        arquivo.write(bloco)


def carregar_estado_monitoramento():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as arquivo:
            estado = json.load(arquivo)
    except (OSError, json.JSONDecodeError):
        return {}
    return estado if isinstance(estado, dict) else {}


def salvar_estado_monitoramento(grupo_atual, canais_teste):
    estado = {
        "grupo_atual": grupo_atual,
        "canais_teste": canais_teste,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as arquivo:
            json.dump(estado, arquivo, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[AVISO] Nao foi possivel salvar estado: {e}")


def reiniciar_processo():
    print("[RESTART] Reiniciando processo do agente...")
    time.sleep(PAUSA_REINICIO_PROCESSO_SEG)
    os.execv(sys.executable, [sys.executable] + sys.argv)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class DetectorCelularFuncionario:
    def __init__(self, canal):
        self.canal = canal
        self.frame_num = 0

    def pessoa_valida(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        altura, largura = frame.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return False

        area_ratio = (bw * bh) / float(largura * altura)
        aspecto = bh / float(bw)

        if area_ratio < 0.002 or area_ratio > 0.65:
            return False
        if aspecto < 0.65 or aspecto > 4.2:
            return False
        return True

    def _aplicar_clahe(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIPL, tileGridSize=(8, 8))
        return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b_ch]), cv2.COLOR_LAB2BGR)

    def _inferir(self, frame_proc, offset_x=0, offset_y=0):
        """Roda YOLO em frame_proc e devolve (pessoas, celulares) com coords ajustadas."""
        with yolo_lock:
            results = modelo_yolo(
                frame_proc,
                classes=[0, 67],
                conf=CONF_CELULAR,
                imgsz=YOLO_IMGSZ,
                iou=0.45,
                verbose=False,
            )
        pessoas, celulares = [], []
        for resultado in results:
            for box in resultado.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 += offset_x; x2 += offset_x
                y1 += offset_y; y2 += offset_y
                conf = float(box.conf[0])
                classe = int(box.cls[0])
                item = {"bbox": (x1, y1, x2, y2), "conf": conf}
                if classe == 0 and conf >= CONF_PESSOA:
                    pessoas.append(item)
                elif classe == 67 and conf >= CONF_CELULAR:
                    celulares.append(item)
        return pessoas, celulares

    def _nms_pessoas(self, candidatos, frame):
        """Deduplica por IoU e aplica filtro pessoa_valida."""
        candidatos.sort(key=lambda x: -x["conf"])
        keep = []
        for c in candidatos:
            if not self.pessoa_valida(frame, c["bbox"]):
                continue
            if all(self.iou(c["bbox"], k["bbox"]) < 0.5 for k in keep):
                keep.append(c)
        return keep

    def detectar_objetos(self, frame, frame_clahe=None):
        if not YOLO_DISPONIVEL or modelo_yolo is None:
            return [], []

        h, w = frame.shape[:2]
        proc = frame_clahe if frame_clahe is not None else self._aplicar_clahe(frame)

        # Frame completo
        p_all, c_all = self._inferir(proc)

        # Crop esquerdo e direito com 1/3 de sobreposição
        cut_l = w * 2 // 3
        cut_r = w // 3
        p_l, c_l = self._inferir(proc[:, :cut_l], offset_x=0)
        p_r, c_r = self._inferir(proc[:, cut_r:], offset_x=cut_r)

        pessoas = self._nms_pessoas(p_all + p_l + p_r, frame)

        # Celulares: deduplicar por IoU simples
        cel_raw = c_all + c_l + c_r
        cel_raw.sort(key=lambda x: -x["conf"])
        celulares = []
        for c in cel_raw:
            if all(self.iou(c["bbox"], k["bbox"]) < 0.5 for k in celulares):
                celulares.append(c)

        if USAR_CROP_CELULAR:
            celulares.extend(self.detectar_celulares_em_pessoas(frame, pessoas))
        return pessoas, celulares

    def detectar_poses(self, frame, frame_clahe=None):
        if not USAR_POSE_MAO_ROSTO or modelo_pose is None:
            return []

        proc = frame_clahe if frame_clahe is not None else self._aplicar_clahe(frame)
        with yolo_lock:
            results = modelo_pose(proc, conf=CONF_PESSOA, imgsz=POSE_IMGSZ, verbose=False)

        poses = []
        for resultado in results:
            if resultado.boxes is None or resultado.keypoints is None:
                continue

            boxes = resultado.boxes.xyxy.cpu().numpy()
            keypoints_xy = resultado.keypoints.xy.cpu().numpy()
            keypoints_conf = None
            if resultado.keypoints.conf is not None:
                keypoints_conf = resultado.keypoints.conf.cpu().numpy()

            for idx, box in enumerate(boxes):
                bbox = tuple(map(int, box))
                if not self.pessoa_valida(frame, bbox):
                    continue
                poses.append(
                    {
                        "bbox": bbox,
                        "xy": keypoints_xy[idx],
                        "conf": keypoints_conf[idx] if keypoints_conf is not None else None,
                    }
                )
        return poses

    def iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def pose_da_pessoa(self, pessoa_bbox, poses):
        melhor_pose = None
        melhor_iou = 0.0
        for pose in poses:
            score = self.iou(pessoa_bbox, pose["bbox"])
            if score > melhor_iou:
                melhor_iou = score
                melhor_pose = pose
        return melhor_pose if melhor_iou >= 0.15 else None

    def keypoint_valido(self, pose, indice, min_conf=0.25):
        if pose is None:
            return None
        if pose["conf"] is not None and pose["conf"][indice] < min_conf:
            return None
        x, y = pose["xy"][indice]
        if x <= 0 and y <= 0:
            return None
        return np.array([x, y], dtype=np.float32)

    def mao_perto_do_rosto(self, pessoa_bbox, pose):
        if pose is None:
            return False, 0.0

        pontos_rosto = [
            self.keypoint_valido(pose, 0, min_conf=MIN_CONF_KEYPOINT_ROSTO),
            self.keypoint_valido(pose, 1, min_conf=MIN_CONF_KEYPOINT_ROSTO),
            self.keypoint_valido(pose, 2, min_conf=MIN_CONF_KEYPOINT_ROSTO),
            self.keypoint_valido(pose, 3, min_conf=MIN_CONF_KEYPOINT_ROSTO),
            self.keypoint_valido(pose, 4, min_conf=MIN_CONF_KEYPOINT_ROSTO),
        ]
        pontos_rosto = [p for p in pontos_rosto if p is not None]
        pulsos = [
            self.keypoint_valido(pose, 9,  min_conf=MIN_CONF_KEYPOINT_ROSTO),
            self.keypoint_valido(pose, 10, min_conf=MIN_CONF_KEYPOINT_ROSTO),
        ]
        pulsos = [p for p in pulsos if p is not None]

        if not pontos_rosto or not pulsos:
            return False, 0.0

        px1, py1, px2, py2 = pessoa_bbox
        escala = max(px2 - px1, py2 - py1)
        limite = max(35.0, escala * 0.28)

        menor_dist = min(float(np.linalg.norm(pulso - rosto)) for pulso in pulsos for rosto in pontos_rosto)
        score = max(0.0, 1.0 - (menor_dist / limite))
        return menor_dist <= limite, score

    def detectar_celulares_em_pessoas(self, frame, pessoas):
        celulares = []
        altura, largura = frame.shape[:2]

        for pessoa in pessoas:
            x1, y1, x2, y2 = pessoa["bbox"]
            bw, bh = x2 - x1, y2 - y1
            if bw < 30 or bh < 50:
                continue

            margem_x = int(bw * 0.15)
            margem_y = int(bh * 0.12)
            rx1 = max(0, x1 - margem_x)
            ry1 = max(0, y1 - margem_y)
            rx2 = min(largura - 1, x2 + margem_x)
            ry2 = min(altura - 1, y2 + margem_y)
            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size == 0:
                continue

            with yolo_lock:
                results = modelo_yolo(
                    crop,
                    classes=[67],
                    conf=CONF_CELULAR,
                    imgsz=YOLO_CELULAR_CROP_IMGSZ,
                    verbose=False,
                )

            for resultado in results:
                for box in resultado.boxes:
                    cx1, cy1, cx2, cy2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    celulares.append(
                        {
                            "bbox": (rx1 + cx1, ry1 + cy1, rx1 + cx2, ry1 + cy2),
                            "conf": conf,
                        }
                    )

        return celulares

    def proporcao_cor_tronco(self, frame, bbox, hsv_min, hsv_max):
        x1, y1, x2, y2 = bbox
        altura, largura = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(largura - 1, x2), min(altura - 1, y2)
        bw, bh = x2 - x1, y2 - y1

        if bw < 20 or bh < 40:
            return 0.0

        tx1 = x1 + int(bw * 0.15)
        tx2 = x2 - int(bw * 0.15)
        ty1 = y1 + int(bh * 0.25)
        ty2 = y1 + int(bh * 0.70)
        tronco = frame[ty1:ty2, tx1:tx2]

        if tronco.size == 0:
            return 0.0

        hsv = cv2.cvtColor(tronco, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_min, hsv_max)
        return float(cv2.countNonZero(mask)) / float(mask.size)

    def centro_pessoa(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def esta_em_area_restrita_branco(self, bbox):
        if self.canal not in CAMERAS_RESTRITAS_BRANCO and self.canal not in ZONAS_RESTRITAS_BRANCO:
            return False

        zonas = ZONAS_RESTRITAS_BRANCO.get(self.canal)
        if not zonas:
            return self.canal in CAMERAS_RESTRITAS_BRANCO

        centro = self.centro_pessoa(bbox)
        for zona in zonas:
            poligono = np.array(zona, dtype=np.int32)
            if cv2.pointPolygonTest(poligono, centro, False) >= 0:
                return True
        return False

    def area_consumo_mercadoria(self, frame, bbox):
        area = AREAS_CONSUMO_MERCADORIA.get(self.canal)
        if area is None:
            return None

        altura, largura = frame.shape[:2]
        zx1, zy1, zx2, zy2 = area["zona"]
        ax1, ay1 = int(largura * zx1), int(altura * zy1)
        ax2, ay2 = int(largura * zx2), int(altura * zy2)
        cx, cy = self.centro_pessoa(bbox)

        if ax1 <= cx <= ax2 and ay1 <= cy <= ay2:
            return area["nome"]
        return None

    def classificar_uniforme(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        alt, larg = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(larg - 1, x2), min(alt - 1, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < 20 or bh < 40:
            return None, 0.0

        tx1 = x1 + int(bw * 0.15)
        tx2 = x2 - int(bw * 0.15)
        ty1 = y1 + int(bh * 0.25)
        ty2 = y1 + int(bh * 0.70)
        tronco = frame[ty1:ty2, tx1:tx2]
        if tronco.size == 0:
            return None, 0.0

        # Única conversão HSV para os dois ranges
        hsv = cv2.cvtColor(tronco, cv2.COLOR_BGR2HSV)
        mask_azul = cv2.inRange(hsv, AZUL_UNIFORME_HSV_MIN, AZUL_UNIFORME_HSV_MAX)
        azul = float(cv2.countNonZero(mask_azul)) / float(mask_azul.size)
        if azul >= PROPORCAO_MIN_AZUL_TRONCO:
            return "azul", azul

        mask_branco = cv2.inRange(hsv, BRANCO_UNIFORME_HSV_MIN, BRANCO_UNIFORME_HSV_MAX)
        branco = float(cv2.countNonZero(mask_branco)) / float(mask_branco.size)
        if branco >= PROPORCAO_MIN_BRANCO_TRONCO and self.esta_em_area_restrita_branco(bbox):
            return "branco_restrito", branco

        return None, max(azul, branco)

    def estimar_boca(self, pose, pessoa_bbox):
        nariz = self.keypoint_valido(pose, 0, min_conf=0.20)
        if nariz is None:
            return None
        px1, py1, px2, py2 = pessoa_bbox
        offset = (py2 - py1) * 0.09
        return np.array([nariz[0], nariz[1] + offset], dtype=np.float32)

    def mao_perto_da_boca(self, pessoa_bbox, pose):
        if pose is None:
            return False, 0.0
        boca = self.estimar_boca(pose, pessoa_bbox)
        if boca is None:
            return False, 0.0
        pulsos = [
            self.keypoint_valido(pose, 9, min_conf=0.25),
            self.keypoint_valido(pose, 10, min_conf=0.25),
        ]
        pulsos = [p for p in pulsos if p is not None]
        if not pulsos:
            return False, 0.0
        px1, py1, px2, py2 = pessoa_bbox
        escala = max(px2 - px1, py2 - py1)
        limite = max(28.0, escala * LIMITE_BOCA_PROPORCAO)
        menor_dist = min(float(np.linalg.norm(p - boca)) for p in pulsos)
        score = max(0.0, 1.0 - (menor_dist / limite))
        return menor_dist <= limite, score

    def braco_levantado_para_boca(self, pose):
        if pose is None:
            return False
        for ombro_idx, cotovelo_idx, pulso_idx in [(5, 7, 9), (6, 8, 10)]:
            ombro = self.keypoint_valido(pose, ombro_idx, min_conf=0.20)
            cotovelo = self.keypoint_valido(pose, cotovelo_idx, min_conf=0.20)
            pulso = self.keypoint_valido(pose, pulso_idx, min_conf=0.20)
            if ombro is None or pulso is None:
                continue
            if pulso[1] >= ombro[1]:
                continue
            if cotovelo is not None:
                v1 = cotovelo - ombro
                v2 = pulso - cotovelo
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1 and n2 > 1:
                    cos_a = np.dot(v1, v2) / (n1 * n2)
                    angulo = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
                    if angulo > 40:
                        return True
            else:
                return True
        return False

    def celular_proximo(self, pessoa_bbox, celular_bbox):
        px1, py1, px2, py2 = pessoa_bbox
        cx1, cy1, cx2, cy2 = celular_bbox
        pw, ph = px2 - px1, py2 - py1

        margem_x = int(pw * 0.35)
        margem_y = int(ph * 0.20)
        ax1 = px1 - margem_x
        ay1 = py1 - margem_y
        ax2 = px2 + margem_x
        ay2 = py2 + margem_y

        centro_x = (cx1 + cx2) // 2
        centro_y = (cy1 + cy2) // 2
        return ax1 <= centro_x <= ax2 and ay1 <= centro_y <= ay2

    def analisar(self, frame):
        # CLAHE compartilhado: aplicar uma vez e reutilizar nas duas inferências
        frame_clahe = self._aplicar_clahe(frame) if YOLO_DISPONIVEL else None
        pessoas, celulares = self.detectar_objetos(frame, frame_clahe=frame_clahe)
        # Só roda pose se houver pessoas — economiza uma inferência de modelo inteira
        poses = self.detectar_poses(frame, frame_clahe=frame_clahe) if pessoas else []
        evidencias = []
        evidencias_consumo = []
        funcionarios = []

        for pessoa in pessoas:
            pose = self.pose_da_pessoa(pessoa["bbox"], poses)
            mao_rosto, score_mao_rosto = self.mao_perto_do_rosto(pessoa["bbox"], pose)
            mao_boca, score_mao_boca = self.mao_perto_da_boca(pessoa["bbox"], pose)
            braco_levantado = self.braco_levantado_para_boca(pose)

            area_consumo = self.area_consumo_mercadoria(frame, pessoa["bbox"])
            # Consumo: mão na boca + braço levantado + na zona correta
            if (
                area_consumo is not None
                and mao_boca
                and braco_levantado
                and score_mao_boca >= SCORE_MIN_CONSUMO_MAO_BOCA
            ):
                evidencias_consumo.append(
                    {
                        "pessoa": pessoa,
                        "area": area_consumo,
                        "motivo": "mao_boca_area_mercadoria",
                        "score_pose": score_mao_boca,
                    }
                )

            uniforme, proporcao_uniforme = self.classificar_uniforme(frame, pessoa["bbox"])
            if uniforme is None:
                continue

            funcionario = {
                "pessoa": pessoa,
                "uniforme": uniforme,
                "proporcao_uniforme": proporcao_uniforme,
            }
            funcionarios.append(funcionario)

            # Celular por pose: exclui apenas quem tem mão na boca em área de consumo
            if mao_rosto:
                esta_comendo_em_area = (
                    area_consumo is not None
                    and mao_boca
                    and score_mao_boca >= SCORE_MIN_CONSUMO_MAO_BOCA
                )
                if not esta_comendo_em_area:
                    evidencias.append(
                        {
                            "pessoa": pessoa,
                            "celular": None,
                            "uniforme": uniforme,
                            "proporcao_uniforme": proporcao_uniforme,
                            "motivo": "mao_rosto",
                            "score_pose": score_mao_rosto,
                        }
                    )
                continue

            for celular in celulares:
                if self.celular_proximo(pessoa["bbox"], celular["bbox"]):
                    evidencias.append(
                        {
                            "pessoa": pessoa,
                            "celular": celular,
                            "uniforme": uniforme,
                            "proporcao_uniforme": proporcao_uniforme,
                            "motivo": "celular_yolo",
                            "score_pose": 0.0,
                        }
                    )
                    break

        return {
            "evidencia": bool(evidencias or evidencias_consumo),
            "evidencia_celular": bool(evidencias),
            "evidencia_consumo": bool(evidencias_consumo),
            "pessoas": pessoas,
            "celulares": celulares,
            "funcionarios": funcionarios,
            "poses": poses,
            "evidencias": evidencias,
            "evidencias_consumo": evidencias_consumo,
        }


# ---------------------------------------------------------------------------
# Evidencias
# ---------------------------------------------------------------------------
def desenhar_deteccoes(frame, resultado):
    anotado = frame.copy()
    if not resultado:
        return anotado

    for pessoa in resultado["pessoas"]:
        x1, y1, x2, y2 = pessoa["bbox"]
        cv2.rectangle(anotado, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            anotado,
            f"pessoa {pessoa['conf']:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 180, 255),
            1,
        )

    for celular in resultado["celulares"]:
        x1, y1, x2, y2 = celular["bbox"]
        cv2.rectangle(anotado, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(
            anotado,
            f"cel {celular['conf']:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1,
        )

    for funcionario in resultado["funcionarios"]:
        x1, y1, x2, y2 = funcionario["pessoa"]["bbox"]
        cv2.rectangle(anotado, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(
            anotado,
            f"{funcionario['uniforme']} {funcionario['proporcao_uniforme']:.0%}",
            (x1, min(anotado.shape[0] - 8, y2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 180, 0),
            1,
        )

    for pose in resultado.get("poses", []):
        for indice in [0, 1, 2, 3, 4, 9, 10]:
            if pose["conf"] is not None and pose["conf"][indice] < 0.25:
                continue
            x, y = pose["xy"][indice]
            if x > 0 or y > 0:
                cv2.circle(anotado, (int(x), int(y)), 4, (0, 255, 0), -1)

    for evidencia in resultado["evidencias"]:
        x1, y1, x2, y2 = evidencia["pessoa"]["bbox"]
        cv2.rectangle(anotado, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            anotado,
            f"{evidencia['uniforme']} {evidencia['proporcao_uniforme']:.0%} + {evidencia['motivo']}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    for evidencia in resultado.get("evidencias_consumo", []):
        x1, y1, x2, y2 = evidencia["pessoa"]["bbox"]
        cv2.rectangle(anotado, (x1, y1), (x2, y2), (0, 120, 255), 3)
        cv2.putText(
            anotado,
            f"consumo? {evidencia['area']} + {evidencia['motivo']}",
            (x1, max(42, y1 - 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 120, 255),
            2,
        )

    return anotado


def salvar_debug(frame, canal, resultado, tempos_evidencia):
    if not DEBUG_DETECCOES:
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = DEBUG_DIR / f"debug_cam{canal:02d}_{ts}.jpg"
    anotado = desenhar_deteccoes(frame, resultado)
    texto = (
        f"CAM {canal:02d} P:{len(resultado['pessoas'])} "
        f"F:{len(resultado['funcionarios'])} C:{len(resultado['celulares'])} "
        f"Pose:{len(resultado.get('poses', []))} "
        f"Cel:{len(resultado['evidencias'])}/{tempos_evidencia[TIPO_ALERTA_CELULAR]:.1f}s "
        f"Cons:{len(resultado.get('evidencias_consumo', []))}/{tempos_evidencia[TIPO_ALERTA_CONSUMO]:.1f}s"
    )
    cv2.putText(
        anotado,
        texto,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    cv2.imwrite(str(path), anotado)


def preparar_frame_exibicao(frame, canal, resultado, tempos_evidencia, status, mostrar_botao=False):
    if resultado:
        exibicao = desenhar_deteccoes(frame, resultado)
    else:
        exibicao = frame.copy()

    altura, largura = exibicao.shape[:2]
    escala = min(1.0, 960.0 / largura)
    if escala < 1.0:
        exibicao = cv2.resize(exibicao, (int(largura * escala), int(altura * escala)))

    cv2.rectangle(exibicao, (0, 0), (exibicao.shape[1], 42), (0, 0, 0), -1)
    cor_status = (0, 0, 255) if status == "ALERTA!" else (0, 200, 0)
    cv2.putText(
        exibicao,
        (
            f"CAM {canal:02d} | {status} | "
            f"celular={tempos_evidencia[TIPO_ALERTA_CELULAR]:.1f}s/{TEMPO_CELULAR_ALERTA}s "
            f"consumo={tempos_evidencia[TIPO_ALERTA_CONSUMO]:.1f}s/{TEMPO_CONSUMO_ALERTA}s"
        ),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        cor_status,
        2,
    )

    if mostrar_botao:
        larg = exibicao.shape[1]
        # Botao PROXIMA (azul)
        bx1, by1, bx2, by2 = larg - 158, 5, larg - 5, 37
        cv2.rectangle(exibicao, (bx1, by1), (bx2, by2), (40, 40, 180), -1)
        cv2.rectangle(exibicao, (bx1, by1), (bx2, by2), (130, 130, 255), 2)
        cv2.putText(exibicao, "PROXIMA [N]", (bx1 + 7, by2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        # Botao VOLTAR (verde)
        vx1, vy1, vx2, vy2 = larg - 320, 5, larg - 165, 37
        cv2.rectangle(exibicao, (vx1, vy1), (vx2, vy2), (40, 120, 40), -1)
        cv2.rectangle(exibicao, (vx1, vy1), (vx2, vy2), (100, 220, 100), 2)
        cv2.putText(exibicao, "VOLTAR [P]", (vx1 + 7, vy2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        # Botao TRAVAR — laranja quando travado, cinza quando auto
        tx1, ty1, tx2, ty2 = larg - 448, 5, larg - 327, 37
        travado = CAMERA_TRAVADA[0]
        cor_fundo = (0, 100, 200) if travado else (60, 60, 60)
        cor_borda = (0, 200, 255) if travado else (140, 140, 140)
        label_travar = "TRAVADA [T]" if travado else "TRAVAR  [T]"
        cv2.rectangle(exibicao, (tx1, ty1), (tx2, ty2), cor_fundo, -1)
        cv2.rectangle(exibicao, (tx1, ty1), (tx2, ty2), cor_borda, 2)
        cv2.putText(exibicao, label_travar, (tx1 + 4, ty2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    return exibicao


def mostrar_tela_loading(canal, mensagem="Carregando camera..."):
    if not MOSTRAR_VIDEO:
        return

    janela_nome = "Agente Vigilancia"
    tela = np.zeros((540, 960, 3), dtype=np.uint8)
    cv2.rectangle(tela, (0, 0), (960, 54), (0, 0, 0), -1)
    cv2.putText(
        tela,
        f"CAM {canal:02d} | {mensagem}",
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 200, 255),
        2,
    )
    cv2.putText(
        tela,
        "Aguarde enquanto o stream RTSP conecta",
        (190, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (220, 220, 220),
        2,
    )
    cv2.putText(
        tela,
        "Atalhos: N proxima | P voltar | Q sair",
        (235, 315),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (160, 160, 160),
        1,
    )
    try:
        cv2.namedWindow(janela_nome, cv2.WINDOW_NORMAL)
        cv2.imshow(janela_nome, tela)
        cv2.waitKey(1)
    except cv2.error:
        pass


def salvar_frame(frame, canal, descricao, tipo=TIPO_ALERTA_CELULAR):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefixo = "consumo" if tipo == TIPO_ALERTA_CONSUMO else "celular"
    path = FRAMES_DIR / f"{prefixo}_cam{canal:02d}_{ts}.jpg"
    anotado = frame.copy()

    cv2.putText(
        anotado,
        f"CAM {canal:02d} | {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        anotado,
        "!! POSSIVEL CONSUMO DE MERCADORIA !!"
        if tipo == TIPO_ALERTA_CONSUMO
        else "!! FUNCIONARIO USANDO CELULAR !!",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        anotado,
        descricao[:90],
        (10, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )

    cv2.imwrite(str(path), anotado)
    return path


def salvar_video(frames_buf, canal, tipo=TIPO_ALERTA_CELULAR):
    if not frames_buf:
        return "sem-video"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefixo = "consumo" if tipo == TIPO_ALERTA_CONSUMO else "celular"
    path = VIDEOS_DIR / f"{prefixo}_cam{canal:02d}_{ts}.avi"
    altura, largura = frames_buf[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(path), fourcc, FRAME_RATE, (largura, altura))

    for frame in frames_buf:
        out.write(frame)

    out.release()
    return path


# ---------------------------------------------------------------------------
# Monitoramento
# ---------------------------------------------------------------------------
def resolver_camera(canal):
    inicio = 1
    for loja in LOJAS_CAMERAS:
        fim = inicio + loja["canais"] - 1
        if inicio <= canal <= fim:
            canal_loja = canal - inicio + 1
            return loja, canal_loja
        inicio = fim + 1
    raise ValueError(f"Canal invalido: {canal}")


def abrir_camera(canal):
    loja, canal_loja = resolver_camera(canal)
    url = loja["rtsp_base"].format(canal_loja)
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, TIMEOUT_RTSP_MS)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, TIMEOUT_RTSP_MS)
    return cap


def monitorar_camera(canal, ultimo_alerta, continuo=False, on_alerta=None):
    mostrar_tela_loading(canal)
    cap = abrir_camera(canal)
    if not cap.isOpened():
        print(f"\r[CAM {canal:02d}] Sem conexao. Pulando...", flush=True)
        mostrar_tela_loading(canal, "Sem conexao. Pulando...")
        time.sleep(0.4)
        return None

    detector = DetectorCelularFuncionario(canal)
    max_buf_pre = int(FRAME_RATE * BUF_VIDEO_SEG)  # calculado antes do bloco inicio
    frames_buf = deque(maxlen=max_buf_pre)
    tempos_evidencia = {
        TIPO_ALERTA_CELULAR: 0.0,
        TIPO_ALERTA_CONSUMO: 0.0,
    }
    frames_guardados = {
        TIPO_ALERTA_CELULAR: None,
        TIPO_ALERTA_CONSUMO: None,
    }
    descricoes_guardadas = {
        TIPO_ALERTA_CELULAR: "",
        TIPO_ALERTA_CONSUMO: "",
    }
    ultimo_resultado = None
    ultimo_yolo_t = None
    ultimo_debug_t = 0.0
    janela_nome = "Agente Vigilancia"
    parar_por_janela = False
    pular_camera_flag = [False]
    voltar_camera_flag = [False]
    btn_bounds = [None]
    btn_voltar_bounds = [None]
    btn_travar_bounds = [None]
    latest_frame = None
    ultimo_frame_recebido_t = time.time()
    latest_lock = Lock()
    parar_captura = False

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if btn_bounds[0]:
            bx1, by1, bx2, by2 = btn_bounds[0]
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                pular_camera_flag[0] = True
        if btn_voltar_bounds[0]:
            vx1, vy1, vx2, vy2 = btn_voltar_bounds[0]
            if vx1 <= x <= vx2 and vy1 <= y <= vy2:
                voltar_camera_flag[0] = True
        if btn_travar_bounds[0]:
            tx1, ty1, tx2, ty2 = btn_travar_bounds[0]
            if tx1 <= x <= tx2 and ty1 <= y <= ty2:
                CAMERA_TRAVADA[0] = not CAMERA_TRAVADA[0]
                estado = "TRAVADA" if CAMERA_TRAVADA[0] else "AUTO"
                print(f"\n[CAM {canal:02d}] Modo {estado}")

    def capturar_frames():
        nonlocal latest_frame, ultimo_frame_recebido_t, parar_captura
        while not parar_captura and not STOP_EVENT.is_set():
            try:
                ret, capturado = cap.read()
            except cv2.error:
                if not parar_captura:
                    time.sleep(0.05)
                continue

            if not ret:
                time.sleep(0.05)
                continue
            with latest_lock:
                latest_frame = capturado
                ultimo_frame_recebido_t = time.time()

    inicio = time.time()
    intervalo = 1.0 / FRAME_RATE
    ultimo_buf_t = 0.0
    ultima_ia_t = 0.0
    inicio_future_ia_t = None
    future_ia = None
    frame_anterior_movimento = None

    if MOSTRAR_VIDEO:
        try:
            cv2.namedWindow(janela_nome, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(janela_nome, on_mouse)
        except cv2.error:
            pass

    thread_captura = Thread(target=capturar_frames, daemon=True)
    thread_captura.start()
    executor_ia = ThreadPoolExecutor(max_workers=1)

    def alerta_pode_disparar(tipo):
        chave = (tipo, canal)
        return time.time() - ultimo_alerta.get(chave, 0) >= COOLDOWN_ALERTA_SEG

    def registrar_disparo(tipo):
        ultimo_alerta[(tipo, canal)] = time.time()

    def montar_alerta(tipo):
        frame_path = salvar_frame(
            frames_guardados[tipo],
            canal,
            descricoes_guardadas[tipo],
            tipo,
        )
        video_path = salvar_video(frames_buf, canal, tipo)
        registrar_disparo(tipo)
        alerta = {
            "tipo": tipo,
            "canal": canal,
            "descricao": descricoes_guardadas[tipo],
            "frame": frame_path,
            "video": video_path,
        }
        tempos_evidencia[tipo] = 0.0
        frames_guardados[tipo] = None
        descricoes_guardadas[tipo] = ""
        return alerta

    try:
        while (continuo or CAMERA_TRAVADA[0] or time.time() - inicio < SEGUNDOS_GRUPO) and not STOP_EVENT.is_set():
            agora = time.time()
            with latest_lock:
                frame = latest_frame.copy() if latest_frame is not None else None
                segundos_sem_frame = agora - ultimo_frame_recebido_t

            if frame is None:
                if segundos_sem_frame >= SEM_FRAME_REINICIAR_SEG:
                    raise CameraTravada(
                        f"CAM {canal:02d} sem frames por {segundos_sem_frame:.1f}s"
                    )
                time.sleep(0.01)
                continue

            if segundos_sem_frame >= SEM_FRAME_REINICIAR_SEG:
                raise CameraTravada(
                    f"CAM {canal:02d} captura congelada por {segundos_sem_frame:.1f}s"
                )

            if (
                future_ia is not None
                and inicio_future_ia_t is not None
                and agora - inicio_future_ia_t >= IA_TRAVADA_REINICIAR_SEG
            ):
                raise CameraTravada(
                    f"CAM {canal:02d} IA travada por {agora - inicio_future_ia_t:.1f}s"
                )

            if agora - ultimo_buf_t >= intervalo:
                frames_buf.append(frame.copy())  # deque com maxlen descarta automaticamente
                ultimo_buf_t = agora

            if future_ia is not None and future_ia.done():
                try:
                    resultado = future_ia.result()
                except Exception as e:
                    print(f"\n[CAM {canal:02d}] Erro na IA: {e}")
                    resultado = None
                future_ia = None
                inicio_future_ia_t = None

                if resultado is not None and ultimo_yolo_t is not None:
                    delta = min(agora - ultimo_yolo_t, 3.0)

                    if resultado["evidencia_celular"]:
                        tempos_evidencia[TIPO_ALERTA_CELULAR] += delta
                        frames_guardados[TIPO_ALERTA_CELULAR] = desenhar_deteccoes(frame, resultado)
                        melhor = max(ev["proporcao_uniforme"] for ev in resultado["evidencias"])
                        uniformes = sorted({ev["uniforme"] for ev in resultado["evidencias"]})
                        motivos = sorted({ev["motivo"] for ev in resultado["evidencias"]})
                        descricoes_guardadas[TIPO_ALERTA_CELULAR] = (
                            f"evidencia={tempos_evidencia[TIPO_ALERTA_CELULAR]:.1f}s/{SEGUNDOS_GRUPO}s | "
                            f"uniforme={','.join(uniformes)} | motivo={','.join(motivos)} | "
                            f"confianca_cor={melhor:.0%}"
                        )
                    else:
                        # Decaimento simetrico para celular (igual ao consumo)
                        tempos_evidencia[TIPO_ALERTA_CELULAR] = max(
                            0.0,
                            tempos_evidencia[TIPO_ALERTA_CELULAR]
                            - (delta * DECAIMENTO_CELULAR_SEM_EVIDENCIA),
                        )
                        if tempos_evidencia[TIPO_ALERTA_CELULAR] <= 0:
                            frames_guardados[TIPO_ALERTA_CELULAR] = None
                            descricoes_guardadas[TIPO_ALERTA_CELULAR] = ""

                    if resultado["evidencia_consumo"]:
                        tempos_evidencia[TIPO_ALERTA_CONSUMO] += delta
                        frames_guardados[TIPO_ALERTA_CONSUMO] = desenhar_deteccoes(frame, resultado)
                        areas = sorted({ev["area"] for ev in resultado["evidencias_consumo"]})
                        scores = [ev["score_pose"] for ev in resultado["evidencias_consumo"]]
                        motivos = sorted({ev["motivo"] for ev in resultado["evidencias_consumo"]})
                        descricoes_guardadas[TIPO_ALERTA_CONSUMO] = (
                            f"evidencia={tempos_evidencia[TIPO_ALERTA_CONSUMO]:.1f}s/{SEGUNDOS_GRUPO}s | "
                            f"area={','.join(areas)} | motivo={','.join(motivos)} | "
                            f"confianca_pose={max(scores):.0%}"
                        )
                    else:
                        tempos_evidencia[TIPO_ALERTA_CONSUMO] = max(
                            0.0,
                            tempos_evidencia[TIPO_ALERTA_CONSUMO]
                            - (delta * DECAIMENTO_CONSUMO_SEM_EVIDENCIA),
                        )
                        if tempos_evidencia[TIPO_ALERTA_CONSUMO] <= 0:
                            frames_guardados[TIPO_ALERTA_CONSUMO] = None
                            descricoes_guardadas[TIPO_ALERTA_CONSUMO] = ""

                if resultado is not None:
                    ultimo_yolo_t = agora
                    ultimo_resultado = resultado

                    tem_algo_para_debug = (
                        resultado["pessoas"] or resultado["funcionarios"] or resultado["celulares"]
                    )
                    if DEBUG_DETECCOES and tem_algo_para_debug and agora - ultimo_debug_t >= DEBUG_INTERVALO_SEG:
                        salvar_debug(frame, canal, resultado, tempos_evidencia)
                        ultimo_debug_t = agora

            if future_ia is None and agora - ultima_ia_t >= IA_INTERVALO_SEG:
                if tem_movimento(frame, frame_anterior_movimento):
                    future_ia = executor_ia.submit(detector.analisar, frame.copy())
                    inicio_future_ia_t = agora
                frame_anterior_movimento = frame
                ultima_ia_t = agora

            if ultimo_resultado:
                yolo_str = (
                    f"P:{len(ultimo_resultado['pessoas'])} "
                    f"F:{len(ultimo_resultado['funcionarios'])} "
                    f"C:{len(ultimo_resultado['celulares'])} "
                    f"Pose:{len(ultimo_resultado.get('poses', []))}"
                )
            else:
                yolo_str = "YOLO:OFF" if not YOLO_DISPONIVEL else "YOLO:..."

            status = (
                "ALERTA!"
                if (
                    tempos_evidencia[TIPO_ALERTA_CELULAR] >= TEMPO_CELULAR_ALERTA
                    or tempos_evidencia[TIPO_ALERTA_CONSUMO] >= TEMPO_CONSUMO_ALERTA
                )
                else "Normal"
            )
            print(
                (
                    f"\r[CAM {canal:02d}] {yolo_str} | "
                    f"celular={tempos_evidencia[TIPO_ALERTA_CELULAR]:4.1f}s | "
                    f"consumo={tempos_evidencia[TIPO_ALERTA_CONSUMO]:4.1f}s | {status}   "
                ),
                end="",
                flush=True,
            )

            if (
                on_alerta is not None
                and tempos_evidencia[TIPO_ALERTA_CELULAR] >= TEMPO_CELULAR_ALERTA
                and frames_guardados[TIPO_ALERTA_CELULAR] is not None
                and alerta_pode_disparar(TIPO_ALERTA_CELULAR)
            ):
                on_alerta(montar_alerta(TIPO_ALERTA_CELULAR))

            if (
                on_alerta is not None
                and tempos_evidencia[TIPO_ALERTA_CONSUMO] >= TEMPO_CONSUMO_ALERTA
                and frames_guardados[TIPO_ALERTA_CONSUMO] is not None
                and alerta_pode_disparar(TIPO_ALERTA_CONSUMO)
            ):
                on_alerta(montar_alerta(TIPO_ALERTA_CONSUMO))

            if MOSTRAR_VIDEO:
                exibicao = preparar_frame_exibicao(
                    frame, canal, ultimo_resultado, tempos_evidencia, status,
                    mostrar_botao=not continuo,
                )
                if not continuo:
                    larg = exibicao.shape[1]
                    btn_bounds[0] = (larg - 158, 5, larg - 5, 37)
                    btn_voltar_bounds[0] = (larg - 320, 5, larg - 165, 37)
                    btn_travar_bounds[0] = (larg - 448, 5, larg - 327, 37)
                cv2.imshow(janela_nome, exibicao)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord("q"):
                    parar_por_janela = True
                    STOP_EVENT.set()
                    break
                if tecla == ord("t"):
                    CAMERA_TRAVADA[0] = not CAMERA_TRAVADA[0]
                    estado = "TRAVADA" if CAMERA_TRAVADA[0] else "AUTO"
                    print(f"\n[CAM {canal:02d}] Modo {estado}")
                if tecla == ord("n") or pular_camera_flag[0]:
                    PULAR_DIRECAO[0] = 1
                    print(f"\n[CAM {canal:02d}] Avancando para proxima camera...")
                    break
                if tecla == ord("p") or voltar_camera_flag[0]:
                    PULAR_DIRECAO[0] = -1
                    print(f"\n[CAM {canal:02d}] Voltando para camera anterior...")
                    break
                if cv2.getWindowProperty(janela_nome, cv2.WND_PROP_VISIBLE) < 1:
                    # Janela fechada pelo usuario: reabre na proxima camera, sem encerrar o programa
                    cv2.destroyWindow(janela_nome)
                    try:
                        cv2.namedWindow(janela_nome, cv2.WINDOW_NORMAL)
                        cv2.setMouseCallback(janela_nome, on_mouse)
                    except cv2.error:
                        pass

            time.sleep(0.02 if future_ia is not None else 0.005)
    finally:
        parar_captura = True
        thread_captura.join(timeout=1.0)
        cap.release()
        executor_ia.shutdown(wait=False, cancel_futures=True)
        if MOSTRAR_VIDEO and (STOP_EVENT.is_set() or parar_por_janela):
            try:
                cv2.destroyWindow(janela_nome)
            except cv2.error:
                pass

    if (
        tempos_evidencia[TIPO_ALERTA_CELULAR] >= TEMPO_CELULAR_ALERTA
        and frames_guardados[TIPO_ALERTA_CELULAR] is not None
        and alerta_pode_disparar(TIPO_ALERTA_CELULAR)
    ):
        return montar_alerta(TIPO_ALERTA_CELULAR)

    if (
        tempos_evidencia[TIPO_ALERTA_CONSUMO] >= TEMPO_CONSUMO_ALERTA
        and frames_guardados[TIPO_ALERTA_CONSUMO] is not None
        and alerta_pode_disparar(TIPO_ALERTA_CONSUMO)
    ):
        return montar_alerta(TIPO_ALERTA_CONSUMO)

    if STOP_EVENT.is_set():
        raise KeyboardInterrupt

    return None


def monitorar_video_local(video_path, canal=0):
    """Processa um arquivo de video local para testar a deteccao. Passe o caminho como argumento."""
    tem_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERRO] Nao foi possivel abrir: {video_path}")
        return

    detector = DetectorCelularFuncionario(canal)
    tempos_evidencia = {TIPO_ALERTA_CELULAR: 0.0, TIPO_ALERTA_CONSUMO: 0.0}
    fps_video = cap.get(cv2.CAP_PROP_FPS) or FRAME_RATE
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    intervalo_frames = max(1, int(fps_video / FRAME_RATE))
    frame_idx = 0
    tempo_video = 0.0
    ultimo_resultado = None
    janela = f"TESTE LOCAL - {Path(video_path).name}"

    print(f"\n[TESTE] {video_path}")
    print(f"[TESTE] FPS: {fps_video:.1f} | frames: {total_frames} | canal simulado: {canal:02d}")
    if tem_display:
        print("[TESTE] Pressione Q para sair")
    else:
        print("[TESTE] Sem display detectado, rodando headless")

    while not STOP_EVENT.is_set():
        ret, frame = cap.read()
        if not ret:
            print("\n[TESTE] Fim do video.")
            break

        frame_idx += 1
        tempo_video = frame_idx / fps_video

        if frame_idx % intervalo_frames != 0:
            continue

        resultado = detector.analisar(frame)
        delta = intervalo_frames / fps_video

        if resultado is not None:
            ultimo_resultado = resultado
            if resultado["evidencia_celular"]:
                tempos_evidencia[TIPO_ALERTA_CELULAR] += min(delta, 3.0)
            if resultado["evidencia_consumo"]:
                tempos_evidencia[TIPO_ALERTA_CONSUMO] += min(delta, 3.0)
            else:
                tempos_evidencia[TIPO_ALERTA_CONSUMO] = max(
                    0.0,
                    tempos_evidencia[TIPO_ALERTA_CONSUMO] - delta * DECAIMENTO_CONSUMO_SEM_EVIDENCIA,
                )

        status = (
            "ALERTA!"
            if (
                tempos_evidencia[TIPO_ALERTA_CELULAR] >= TEMPO_CELULAR_ALERTA
                or tempos_evidencia[TIPO_ALERTA_CONSUMO] >= TEMPO_CONSUMO_ALERTA
            )
            else "Normal"
        )

        if ultimo_resultado:
            r = ultimo_resultado
            print(
                f"\r[{tempo_video:6.1f}s] P:{len(r['pessoas'])} F:{len(r['funcionarios'])} "
                f"C:{len(r['celulares'])} Pose:{len(r.get('poses', []))} "
                f"Cel:{len(r['evidencias'])} Cons:{len(r.get('evidencias_consumo', []))} | "
                f"cel={tempos_evidencia[TIPO_ALERTA_CELULAR]:.1f}s "
                f"cons={tempos_evidencia[TIPO_ALERTA_CONSUMO]:.1f}s | {status}   ",
                end="",
                flush=True,
            )

        if tem_display and MOSTRAR_VIDEO and ultimo_resultado is not None:
            try:
                exibicao = preparar_frame_exibicao(frame, canal, ultimo_resultado, tempos_evidencia, status)
                cv2.imshow(janela, exibicao)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                pass

    cap.release()
    if tem_display:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    print(
        f"\n[TESTE] Resultado final: "
        f"celular={tempos_evidencia[TIPO_ALERTA_CELULAR]:.1f}s/{TEMPO_CELULAR_ALERTA}s | "
        f"consumo={tempos_evidencia[TIPO_ALERTA_CONSUMO]:.1f}s/{TEMPO_CONSUMO_ALERTA}s"
    )
    alertas = []
    if tempos_evidencia[TIPO_ALERTA_CELULAR] >= TEMPO_CELULAR_ALERTA:
        alertas.append(TIPO_ALERTA_CELULAR)
    if tempos_evidencia[TIPO_ALERTA_CONSUMO] >= TEMPO_CONSUMO_ALERTA:
        alertas.append(TIPO_ALERTA_CONSUMO)
    if alertas:
        print(f"[TESTE] !! ALERTAS DETECTADOS: {', '.join(alertas)} !!")
    else:
        print("[TESTE] Nenhum alerta atingido no video.")


def grupos_cameras(canais_especificos=None):
    if canais_especificos:
        canais = canais_especificos
        return [canais[i : i + CAMERAS_GRUPO] for i in range(0, len(canais), CAMERAS_GRUPO)]

    canais = list(range(1, NUM_CAMERAS + 1))
    return [canais[i : i + CAMERAS_GRUPO] for i in range(0, len(canais), CAMERAS_GRUPO)]


def main():
    log_path = setup()
    incidente_count = 0
    ultimo_alerta = {}

    def _canal_do_nome(path):
        m = re.search(r"cam(\d+)", Path(path).stem)
        return int(m.group(1)) if m else 0

    # Aceita arquivos .avi/.mp4 individuais ou um diretório de vídeos
    args_video = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_file() and p.suffix.lower() in (".avi", ".mp4", ".mkv"):
            args_video.append(p)
        elif p.is_dir():
            args_video.extend(sorted(p.glob("*.avi")) + sorted(p.glob("*.mp4")))

    if args_video:
        print(f"\n[TESTE] {len(args_video)} video(s) para processar")
        for vp in args_video:
            if STOP_EVENT.is_set():
                break
            try:
                monitorar_video_local(vp, canal=_canal_do_nome(vp))
            except KeyboardInterrupt:
                STOP_EVENT.set()
        sys.exit(0)

    canais_teste = [int(arg) for arg in sys.argv[1:] if arg.isdigit()]
    canais_teste = [canal for canal in canais_teste if 1 <= canal <= NUM_CAMERAS]
    modo_camera_fixa = len(canais_teste) == 1
    grupos = grupos_cameras(canais_teste)
    estado = carregar_estado_monitoramento()
    estado_canais_teste = estado.get("canais_teste")
    if estado_canais_teste == canais_teste:
        grupo_atual = int(estado.get("grupo_atual", 0)) % len(grupos)
    else:
        grupo_atual = 0

    print("=" * 60)
    print("  AGENTE DE VIGILANCIA LOCAL")
    print("  Sistema iniciado - Ctrl+C para parar")
    print(f"  {NUM_CAMERAS} cameras | {CAMERAS_GRUPO} por vez | {SEGUNDOS_GRUPO}s por grupo")
    print(f"  Alarme celular: {TEMPO_CELULAR_ALERTA}s acumulados em {SEGUNDOS_GRUPO}s")
    print(
        f"  Alarme consumo: {TEMPO_CONSUMO_ALERTA}s acumulados em {SEGUNDOS_GRUPO}s "
        f"| canais {', '.join(str(c) for c in AREAS_CONSUMO_MERCADORIA)}"
    )
    if canais_teste:
        print(f"  Modo teste: somente camera(s) {', '.join(str(c) for c in canais_teste)}")
    if modo_camera_fixa:
        print("  Modo camera fixa: sem troca/reinicio automatico")
    if grupo_atual:
        print(f"  Retomando do grupo salvo: {grupo_atual + 1}/{len(grupos)}")
    print(f"  Log: {log_path}")
    print("=" * 60 + "\n")

    escrever_log(
        log_path,
        f"Sistema iniciado. {NUM_CAMERAS} cameras | {CAMERAS_GRUPO} por vez | "
        f"{SEGUNDOS_GRUPO}s por grupo | celular {TEMPO_CELULAR_ALERTA}s"
        f" | consumo {TEMPO_CONSUMO_ALERTA}s canais "
        f"{','.join(str(c) for c in AREAS_CONSUMO_MERCADORIA)}"
        f"{' | teste ' + ','.join(str(c) for c in canais_teste) if canais_teste else ''}.",
    )

    def registrar_alerta(alerta):
        nonlocal incidente_count
        incidente_count += 1
        escrever_incidente(
            log_path,
            incidente_count,
            alerta["canal"],
            alerta["descricao"],
            alerta["frame"],
            alerta["video"],
            alerta.get("tipo", TIPO_ALERTA_CELULAR),
        )

    reiniciar_apos_finally = False

    try:
        while not STOP_EVENT.is_set():
            grupo = grupos[grupo_atual]
            salvar_estado_monitoramento(grupo_atual, canais_teste)
            print(f"\n[GRUPO] Monitorando cameras: {', '.join(str(c) for c in grupo)}")

            if len(grupo) == 1:
                try:
                    alerta = monitorar_camera(
                        grupo[0],
                        ultimo_alerta,
                        continuo=modo_camera_fixa,
                        on_alerta=registrar_alerta,
                    )
                except CameraTravada as e:
                    escrever_log(log_path, f"{e}. Reiniciando monitoramento da camera.")
                    time.sleep(PAUSA_REINICIO_CAMERA_SEG)
                    continue
                if alerta is not None:
                    registrar_alerta(alerta)
            else:
                with ThreadPoolExecutor(max_workers=CAMERAS_GRUPO) as executor:
                    futures = [executor.submit(monitorar_camera, canal, ultimo_alerta) for canal in grupo]
                    for future in as_completed(futures):
                        if STOP_EVENT.is_set():
                            break
                        try:
                            alerta = future.result()
                        except CameraTravada as e:
                            escrever_log(log_path, f"{e}. Camera sera reaberta no proximo ciclo.")
                            continue
                        if alerta is None:
                            continue

                        incidente_count += 1
                        escrever_incidente(
                            log_path,
                            incidente_count,
                            alerta["canal"],
                            alerta["descricao"],
                            alerta["frame"],
                            alerta["video"],
                            alerta.get("tipo", TIPO_ALERTA_CELULAR),
                        )

            if STOP_EVENT.is_set():
                break

            direcao = PULAR_DIRECAO[0] if PULAR_DIRECAO[0] != 0 else 1
            PULAR_DIRECAO[0] = 0
            grupo_atual = (grupo_atual + direcao) % len(grupos)

    except KeyboardInterrupt:
        STOP_EVENT.set()
    except Exception as e:
        escrever_log(log_path, f"Erro fatal no monitoramento: {e}")
        if REINICIAR_PROCESSO_EM_ERRO and not STOP_EVENT.is_set():
            reiniciar_apos_finally = True
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print("\n\n[STOP] Monitoramento encerrado pelo usuario.")
        print(f"[INFO] Total de incidentes: {incidente_count}")
        print(f"[INFO] Logs  : {log_path}")
        print(f"[INFO] Frames: {FRAMES_DIR}")
        print(f"[INFO] Videos: {VIDEOS_DIR}")
        escrever_log(log_path, f"Sistema encerrado. Total incidentes: {incidente_count}")
        if reiniciar_apos_finally:
            reiniciar_processo()
        sys.exit(0)


if __name__ == "__main__":
    main()
