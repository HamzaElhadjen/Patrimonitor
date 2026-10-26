"""Aide à l'hypothèse interprétable, volontairement conservatrice."""
def suggest_mechanisms(features):
    """features: descripteurs fournis/annotés par l'utilisateur; aucune conclusion automatique."""
    hypotheses = []
    orient = str(features.get("orientation_pattern", "")).lower()
    if "diagonal" in orient or "diagonale" in orient:
        hypotheses.append(("cisaillement", "motif diagonal renseigné", "faible"))
    if "stair" in orient or "escalier" in orient:
        hypotheses.append(("mouvement différentiel possible", "motif en escalier renseigné", "faible"))
    if "vertical" in orient or "verticale" in orient:
        hypotheses.append(("tassement ou retrait à examiner", "motif vertical renseigné", "faible"))
    return {
        "hypotheses": hypotheses,
        "warning": ("AIDE À L'HYPOTHÈSE UNIQUEMENT — aucun diagnostic structurel. "
                    "Faire confirmer par un ingénieur structure/restaurateur."),
        "expert_validation": "Non effectuée"
    }
