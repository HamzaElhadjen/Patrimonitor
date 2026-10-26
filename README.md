# PatriMonitor v4 — base expérimentale

## Ce qui est inclus
- Application PyQt5 de détection simple et comparaison A-B.
- Générateur synthétique reproductible (`synth/simulator.py`) avec masques vérité terrain.
- Briques expérimentales pour jauge virtuelle (`gauge.py`), incertitude (`uncertainty.py`),
  graphe (`crackgraph.py`), fusion environnementale (`envfusion.py`), aide à l'hypothèse
  (`diagnostic.py`) et guidage de re-photographie (`guide.py`).
- Tests unitaires de base.

## Limites importantes
Ces modules sont des prototypes et ne constituent pas une validation scientifique. Le générateur
est procédural simple, pas un simulateur mécanique réaliste. Le score d'incertitude actuel est
heuristique et non calibré. Le graphe fournit actuellement des nœuds et des métriques globales,
pas un appariement fiable segment-à-segment. La jauge virtuelle nécessite des points sélectionnés,
une compensation de mouvement fiable et une échelle métrique avant toute interprétation en mm.
Le diagnostic ne remplace jamais un expert.

Le modèle fourni par l'utilisateur reste nécessaire dans `weights/best.pt`, ou via `PATRI_MODEL`.
Aucune performance, précision, nouveauté ou détectabilité n'est revendiquée sans validation.

## Installation indicative
Python 3.11 recommandé. Installer les dépendances déjà utilisées par l'application :
`ultralytics`, `opencv-python`, `numpy`, `pyvista`, `pyvistaqt`, `PyQt5`, `scikit-image`.
Pour les tests : `pytest`.

## Lancement
`python app.py`

## Générateur
`python -c "from synth.simulator import generate_dataset; generate_dataset('dataset_synthetique', 20, 42)"`

Les données générées sont synthétiques; elles ne doivent pas être mélangées aux données réelles sans
identifier explicitement leur provenance et évaluer le décalage de domaine.

## Protocole d'acquisition multi-vues pour une future reconstruction 3DGS
1. Définir une zone de mur stable et une échelle/cible métrique visible.
2. Photographier en lumière diffuse, exposition et focale aussi constantes que possible.
3. Prévoir un recouvrement élevé entre vues adjacentes (cible pratique initiale : 70–80 %, à valider).
4. Capturer plusieurs distances et angles sans masquer les zones fissurées; éviter flou et reflets.
5. Documenter appareil, focale, date, lumière, distance approximative et identifiant de chaque vue.
6. Vérifier d'abord la reconstruction/poses avec COLMAP; mesurer l'erreur de reprojection avant
   d'entraîner un modèle 3D Gaussian Splatting.
7. Ne pas interpréter une visualisation 3DGS comme une mesure métrique sans échelle et validation.
