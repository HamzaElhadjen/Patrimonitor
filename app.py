# app.py - PatriMonitor v2.1 (SHM, analyse temporelle 2.5D)
from ultralytics import YOLO  # DOIT rester en premier (fix WinError 1114)

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QTextEdit, QVBoxLayout, QWidget,
    QSlider, QCheckBox, QProgressBar, QFrame
)
from pyvistaqt import QtInteractor
from skimage.morphology import skeletonize  # pip install scikit-image
from crackgraph import extract_crack_graph
from uncertainty import local_tolerance_map, compare_masks_with_tolerance
from synth.simulator import generate_dataset

# ---------------- Configuration ----------------
# ---------------- Configuration ----------------
DEFAULT_MODEL = r"C:\Users\hamza\OneDrive\Bureau\italy project\runs\segment\train\weights\best.pt"
MODEL_PATH = Path(os.getenv("PATRI_MODEL", DEFAULT_MODEL))
MM_PER_PX = float(os.getenv("PATRI_MM_PER_PX", 0)) or None  # échelle optionnelle
CONF = 0.10
MIN_INLIERS, MIN_INLIER_RATIO = 15, 0.30   # contrôle qualité du recalage
TOLERANCE_PX = 5                           # tolérance de désalignement résiduel


# ---------------- Logique métier (sans UI) ----------------
def register(ref, img):
    """Aligne img sur ref (SIFT + FLANN + RANSAC). Lève ValueError si fiabilité insuffisante."""
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        raise ValueError("Pas assez de points d'intérêt détectés.")

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    pairs = [p for p in flann.knnMatch(d2, d1, k=2) if len(p) == 2]
    good = [m for m, n in pairs if m.distance < 0.7 * n.distance]
    if len(good) < MIN_INLIERS:
        raise ValueError(f"Recalage impossible : {len(good)} correspondances.")

    p1 = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    p2 = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(p2, p1, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("Homographie non calculable.")

    n_in = int(inl.sum())
    ratio = n_in / len(good)
    det = np.linalg.det(H[:2, :2])  # homographie dégénérée si échelle aberrante
    if n_in < MIN_INLIERS or ratio < MIN_INLIER_RATIO or not 0.5 < det < 2.0:
        raise ValueError(f"Recalage peu fiable (inliers={n_in}, ratio={ratio:.0%}).")

    err = cv2.perspectiveTransform(p2[inl.ravel() == 1], H) - p1[inl.ravel() == 1]
    rmse = float(np.sqrt((err ** 2).sum(-1).mean()))
    h, w = ref.shape[:2]
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_CUBIC), \
        {"matches": len(good), "inliers": n_in, "ratio": ratio, "rmse": rmse}


def binary_mask(result, shape):
    """Résultat YOLO -> masque binaire uint8 (0/255)."""
    m = np.zeros(shape[:2], np.uint8)
    if result.masks is not None:
        for poly in result.masks.xy:
            cv2.fillPoly(m, [np.int32(poly)], 255)
    return m


def measure(mask):
    """Surface (px) et longueur approx. (px) via squelette."""
    area = int((mask > 0).sum())
    length = int(skeletonize(mask > 0).sum()) if area else 0
    return area, length


def build_evolution(ref, m1, m2, tolerance_px=TOLERANCE_PX):
    """Bleu = Date 1, Rouge = Date 2; nouvelle zone hors tolérance choisie."""
    radius = max(1, int(np.ceil(tolerance_px)))
    tol = cv2.dilate(m1, np.ones((2 * radius + 1,) * 2, np.uint8))
    new = cv2.bitwise_and(m2, cv2.bitwise_not(tol))
    layer = np.zeros_like(ref)
    layer[m1 > 0] = (255, 0, 0)
    layer[m2 > 0] = (0, 0, 255)
    blended = cv2.addWeighted(ref, 0.4, layer, 0.6, 0)
    out = ref.copy()
    out[(m1 > 0) | (m2 > 0)] = blended[(m1 > 0) | (m2 > 0)]
    return out, new

QSS = """
QMainWindow, QWidget { background:#1e222b; color:#e6e9ef; font-family:'Segoe UI'; font-size:13px; }
QFrame#side { background:#262b36; border-radius:10px; }
QLabel#title { font-size:18px; font-weight:700; }
QLabel#sec { color:#8b93a7; font-size:11px; font-weight:600; margin-top:8px; }
QLabel#view { background:#14171d; border:1px solid #333a4a; border-radius:8px; }
QPushButton { background:#323847; border:none; border-radius:6px; padding:9px; }
QPushButton:hover { background:#3d4559; }
QPushButton:disabled { background:#2a2e39; color:#5b6377; }
QPushButton#primary { background:#4f8cff; font-weight:600; }
QPushButton#primary:hover { background:#6aa0ff; }
QPushButton#accent { background:#e5683c; font-weight:600; }
QPushButton#accent:hover { background:#f07d54; }
QTextEdit { background:#14171d; border:1px solid #333a4a; border-radius:6px; font-family:Consolas; }
QSlider::groove:horizontal { height:4px; background:#3d4559; }
QSlider::handle:horizontal { background:#4f8cff; width:14px; margin:-6px 0; border-radius:7px; }
QProgressBar { max-height:4px; background:#323847; border:none; }
QProgressBar::chunk { background:#4f8cff; }
"""


def enhance(img):
    """CLAHE sur le canal L : renforce le contraste des micro-fissures."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def paint(img, mask, bgr=(0, 0, 255)):
    out = img.copy()
    layer = np.full_like(img, bgr)
    out[mask > 0] = cv2.addWeighted(img, 0.4, layer, 0.6, 0)[mask > 0]
    return out

class AnalysisWorker(QThread):
    done = pyqtSignal(object, str)
    failed = pyqtSignal(str)

    def __init__(self, model, mode, img1, img2, conf, clahe):
        super().__init__()
        self.model, self.mode, self.img1, self.img2 = model, mode, img1, img2
        self.conf, self.clahe = conf, clahe

    def _seg(self, img):
        src = enhance(img) if self.clahe else img
        r = self.model.predict(src, conf=self.conf, verbose=False)[0]
        return binary_mask(r, img.shape), (0 if r.masks is None else len(r.masks))

    def run(self):
        try:
            u, k = ("mm", MM_PER_PX) if MM_PER_PX else ("px", 1.0)
            if self.mode == "single":
                m, n = self._seg(self.img1)
                area, length = measure(m)
                out = paint(self.img1, m)
                rep = (f"DÉTECTION SIMPLE (conf ≥ {self.conf:.2f})\n"
                       f"Instances : {n}\n"
                       f"Surface : {area * k * k:.0f} {u}²\n"
                       f"Longueur cumulée : {length * k:.0f} {u}")
            else:
                aligned, qc = register(self.img1, self.img2)
                m1, _ = self._seg(self.img1)
                m2, _ = self._seg(aligned)
                # Tolérance adaptée au RMSE global : proxy expérimental, non calibré localement.
                tol_map = local_tolerance_map(m1.shape, qc["rmse"])
                new, uncertainty_info = compare_masks_with_tolerance(m1, m2, tol_map)
                out, _ = build_evolution(
                    self.img1, m1, m2,
                    tolerance_px=uncertainty_info["tolerance_proxy_px"]
                )
                (a1, l1), (a2, l2), (an, _) = measure(m1), measure(m2), measure(new)
                var = (a2 - a1) / a1 * 100 if a1 else float("nan")
                graph_a = extract_crack_graph(m1)
                graph_b = extract_crack_graph(m2)
                rep = (f"COMPARAISON TEMPORELLE (conf ≥ {self.conf:.2f})\\n"
                       f"Recalage : {qc['inliers']}/{qc['matches']} inliers "
                       f"({qc['ratio']:.0%}), RMSE {qc['rmse']:.2f} px\\n"
                       f"A : {a1 * k * k:.0f} {u}² | {l1 * k:.0f} {u}\\n"
                       f"B : {a2 * k * k:.0f} {u}² | {l2 * k:.0f} {u}\\n"
                       f"Variation surface : {var:+.1f} %\\n"
                       f"Nouvelle zone (tolérance {uncertainty_info['tolerance_proxy_px']:.1f} px) : "
                       f"{an * k * k:.0f} {u}²\\n"
                       f"Réseau A : {graph_a['summary']['endpoints']} extrémités, "
                       f"{graph_a['summary']['junctions']} jonctions, squelette "
                       f"{graph_a['summary']['length_px']} px\\n"
                       f"Réseau B : {graph_b['summary']['endpoints']} extrémités, "
                       f"{graph_b['summary']['junctions']} jonctions, squelette "
                       f"{graph_b['summary']['length_px']} px\\n"
                       f"Score heuristique de changement : "
                       f"{uncertainty_info['new_pixel_fraction_heuristic']:.1%} "
                       f"(pas une probabilité calibrée)\\n"
                       f"Significativité statistique : non établie\\n"
                       f"Limite : hypothèse planaire; validation sur données réelles requise.")
            self.done.emit(out, rep)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class PatriMonitorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PatriMonitor v4 — SHM Patrimoine (prototype expérimental)")
        self.setGeometry(40, 40, 1500, 850)
        self.img1 = self.img2 = self.result_bgr = self.worker = None
        self.report_text = ""
        self.model = self._load_model()
        self._init_ui()
        self._init_scene()
        self._refresh()

    def _load_model(self):
        path = MODEL_PATH
        if not path.exists():
            f, _ = QFileDialog.getOpenFileName(self, "Sélectionner best.pt", "", "Poids (*.pt)")
            path = Path(f) if f else None
        return YOLO(str(path)) if path and path.exists() else None

    # ---------- UI ----------
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)

        # Panneau latéral
        side = QFrame(objectName="side")
        side.setFixedWidth(330)
        s = QVBoxLayout(side)
        s.setContentsMargins(16, 16, 16, 16)
        s.addWidget(QLabel("PatriMonitor", objectName="title"))
        s.addWidget(QLabel("Structural Health Monitoring" if self.model else
                           "⚠ Modèle introuvable", objectName="sec"))

        s.addWidget(QLabel("IMAGES", objectName="sec"))
        self.btn1 = QPushButton("Charger image A (unique / référence)")
        self.btn2 = QPushButton("Charger image B (autre date / angle)")
        self.btn1.clicked.connect(lambda: self._load(1))
        self.btn2.clicked.connect(lambda: self._load(2))
        s.addWidget(self.btn1)
        s.addWidget(self.btn2)

        s.addWidget(QLabel("ANALYSE", objectName="sec"))
        self.btn_single = QPushButton("Détecter les fissures (image A)", objectName="primary")
        self.btn_cmp = QPushButton("Comparer A ↔ B (recalage + IA)", objectName="accent")
        self.btn_single.clicked.connect(lambda: self._run("single"))
        self.btn_cmp.clicked.connect(lambda: self._run("compare"))
        s.addWidget(self.btn_single)
        s.addWidget(self.btn_cmp)

        s.addWidget(QLabel("OPTIMISATION", objectName="sec"))
        self.lbl_conf = QLabel()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(5, 60)
        self.slider.setValue(int(CONF * 100))
        self.slider.valueChanged.connect(
            lambda v: self.lbl_conf.setText(f"Seuil de confiance : {v / 100:.2f}"))
        self.slider.valueChanged.emit(self.slider.value())
        self.chk_clahe = QCheckBox("Renforcer le contraste (CLAHE)")
        self.chk_clahe.setChecked(True)
        s.addWidget(self.lbl_conf)
        s.addWidget(self.slider)
        s.addWidget(self.chk_clahe)

        s.addWidget(QLabel("RAPPORT", objectName="sec"))
        self.report = QTextEdit(readOnly=True)
        s.addWidget(self.report, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        s.addWidget(self.progress)

        s.addWidget(QLabel("DONNÉES SYNTHÉTIQUES", objectName="sec"))
        self.btn_synth = QPushButton("Générer un petit jeu synthétique")
        self.btn_synth.clicked.connect(self._generate_synthetic)
        s.addWidget(self.btn_synth)

        row = QHBoxLayout()
        self.btn_export = QPushButton("Exporter")
        self.btn_reset = QPushButton("Réinitialiser")
        self.btn_export.clicked.connect(self._export)
        self.btn_reset.clicked.connect(self._reset)
        row.addWidget(self.btn_export)
        row.addWidget(self.btn_reset)
        s.addLayout(row)
        root.addWidget(side)

        # Zone principale
        main = QVBoxLayout()
        views = QHBoxLayout()
        self.lbl1 = self._view("Image A")
        self.lbl2 = self._view("Image B")
        self.lbl_res = self._view("Résultat")
        for v in (self.lbl1, self.lbl2, self.lbl_res):
            views.addWidget(v)
        main.addLayout(views, 1)
        self.plotter = QtInteractor(self)
        main.addWidget(self.plotter.interactor, 2)
        root.addLayout(main, 1)

    @staticmethod
    def _view(text):
        l = QLabel(text, objectName="view")
        l.setAlignment(Qt.AlignCenter)
        l.setMinimumSize(200, 200)
        return l

    def _init_scene(self):
        self.plotter.set_background("#14171d")
        self.plotter.add_mesh(pv.Text3D("En attente d'analyse", depth=0.1), color="white")
        self.plotter.reset_camera()

    # ---------- Logique UI ----------
    def _refresh(self, busy=False):
        ok = self.model is not None and not busy
        self.btn_single.setEnabled(ok and self.img1 is not None)
        self.btn_cmp.setEnabled(ok and self.img1 is not None and self.img2 is not None)
        self.btn_export.setEnabled(self.result_bgr is not None and not busy)
        self.progress.setVisible(busy)

    def _load(self, idx):
        f, _ = QFileDialog.getOpenFileName(self, f"Image {'AB'[idx - 1]}", "",
                                           "Images (*.png *.jpg *.jpeg)")
        img = cv2.imread(f) if f else None
        if img is None:
            return
        if idx == 1:
            self.img1 = img
            self._show(img, self.lbl1)
        else:
            self.img2 = img
            self._show(img, self.lbl2)
        self._refresh()

    @staticmethod
    def _show(bgr, label):
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        h, w, ch = rgb.shape
        q = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        label.setPixmap(QPixmap.fromImage(q).scaled(
            label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _run(self, mode):
        self._refresh(busy=True)
        self.report.setText("Analyse en cours...")
        self.worker = AnalysisWorker(self.model, mode, self.img1, self.img2,
                                     self.slider.value() / 100, self.chk_clahe.isChecked())
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_fail)
        self.worker.start()

    def _on_done(self, result, report):
        self.result_bgr, self.report_text = result, report
        self.report.setText(report)
        self._show(result, self.lbl_res)
        self._update_3d()
        self._refresh()

    def _on_fail(self, msg):
        self.report.setText(f"Échec : {msg}")
        QMessageBox.warning(self, "Analyse interrompue", msg)
        self._refresh()

    def _update_3d(self):
        self.plotter.clear()
        rgb = cv2.cvtColor(self.result_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        plane = pv.Plane(i_size=w / 100, j_size=h / 100)
        plane.texture_map_to_plane(inplace=True)
        self.plotter.add_mesh(plane, texture=pv.Texture(rgb), name="wall")
        self.plotter.view_xy()
        self.plotter.reset_camera()

    def _generate_synthetic(self):
        """Génère un petit dataset synthétique reproductible; ne remplace pas des données réelles."""
        folder = QFileDialog.getExistingDirectory(self, "Dossier de sortie du jeu synthétique")
        if not folder:
            return
        try:
            metadata = generate_dataset(folder, count=20, seed=42)
            QMessageBox.information(
                self, "Jeu synthétique généré",
                f"{len(metadata)} exemples créés dans :\\n{folder}\\n\\n"
                "Images, masques et metadata.json sont synthétiques. "
                "Ils ne constituent pas une validation sur maçonnerie réelle."
            )
        except Exception as exc:
            QMessageBox.warning(self, "Échec de génération", str(exc))

    def _export(self):
        f, _ = QFileDialog.getSaveFileName(self, "Exporter", "resultat.png", "PNG (*.png)")
        if f:
            cv2.imwrite(f, self.result_bgr)
            Path(f).with_suffix(".txt").write_text(self.report_text, encoding="utf-8")

    def _reset(self):
        self.img1 = self.img2 = self.result_bgr = None
        self.report_text = ""
        self.report.clear()
        for l, t in ((self.lbl1, "Image A"), (self.lbl2, "Image B"), (self.lbl_res, "Résultat")):
            l.clear()
            l.setText(t)
        self.plotter.clear()
        self._init_scene()
        self._refresh()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    win = PatriMonitorApp()
    win.show()
    sys.exit(app.exec_())
