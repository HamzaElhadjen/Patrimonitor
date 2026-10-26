# Protocole d'acquisition multi-vues avant 3D Gaussian Splatting

Cette piste n'est pas intégrée comme reconstruction 3D automatique dans l'application :
aucune donnée multi-vues n'est disponible actuellement. Procédure initiale proposée :

1. Choisir une zone stable, baliser la zone et inclure une cible métrique ou une distance connue.
2. Garder autant que possible le même appareil, la même focale, l'exposition et une lumière diffuse.
3. Capturer une séquence avec recouvrement élevé entre vues adjacentes; 70–80 % est une valeur de départ pratique à tester, pas une garantie.
4. Inclure plusieurs angles et distances, sans flou, reflets importants ni occultation des fissures.
5. Enregistrer les métadonnées : appareil, focale, date/heure, conditions d'éclairage et ordre des vues.
6. Exécuter COLMAP/pycolmap et examiner les poses et l'erreur de reprojection avant tout entraînement 3DGS.
7. Pour comparer deux campagnes, prévoir des points de contrôle stables ou une zone stable pour l'alignement, puis mesurer l'erreur.
8. N'affirmer aucune précision métrique sur la seule base du réalisme visuel d'une scène 3DGS.

La segmentation 2D vers 3D, l'alignement inter-campagnes et la validation IoU 3D restent à développer.
