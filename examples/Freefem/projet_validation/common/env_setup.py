"""
Correctif Windows : pyfreefem invoque en interne l'utilitaire Unix `stdbuf`
pour capturer la sortie de FreeFem++ en temps reel. Cet utilitaire n'existe
pas nativement sous Windows, mais Git for Windows en fournit une version
dans son propre dossier (Git\\usr\\bin).

Importer ce module (import common.env_setup, ou depuis un run.py :
from . import env_setup en remontant jusqu'a common) applique le correctif
automatiquement, sans dependre d'une cellule de notebook executee au bon
moment / dans le bon kernel.
"""
import os

# Chemins candidats, dans l'ordre de priorite. Le premier qui existe est
# utilise. Si aucun n'existe, le correctif est silencieusement ignore
# (sur Linux/Mac, stdbuf existe deja nativement -> rien a faire).
_CANDIDATE_PATHS = [
    r"C:\Program Files\Git\usr\bin",
    r"C:\Program Files (x86)\Git\usr\bin",
]


def ensure_stdbuf_available():
    """Ajoute le dossier contenant stdbuf.exe au PATH du processus courant,
    si ce n'est pas deja fait. Sans effet si aucun chemin candidat n'existe
    (ex: sur Linux/Mac) ou si le PATH le contient deja."""
    if os.name != "nt":
        return
    current_path = os.environ.get("PATH", "")
    for candidate in _CANDIDATE_PATHS:
        if candidate in current_path:
            return
        if os.path.isfile(os.path.join(candidate, "stdbuf.exe")):
            os.environ["PATH"] = current_path + os.pathsep + candidate
            return


# Applique le correctif des l'import de ce module.
ensure_stdbuf_available()
