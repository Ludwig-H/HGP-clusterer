import json
import os

# Notebook structure
notebook = {
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "machine_shape": "hm"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python",
      "version": "3.10"
    }
  },
  "cells": []
}

def add_cell(source, cell_type="code", title=None):
    if isinstance(source, list):
        pass
    else:
        source = [line + "\n" for line in source.splitlines()]
        if source and source[-1].endswith("\n"):
             source[-1] = source[-1][:-1]

    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
        
    if title:
        cell["metadata"]["id"] = title
    notebook["cells"].append(cell)

# -----------------------------------------------------------------------------
# 1. Introduction
# -----------------------------------------------------------------------------
add_cell("""# HGP-clusterer : 4D Panoptic Segmentation sur SemanticKITTI

Ce notebook implémente un pipeline de segmentation panoptique 4D en utilisant **HGP-clusterer**.

**Pipeline :**
1.  **Setup** : Installation des dépendances (Geogram/CGAL, HGP-clusterer).
2.  **Data** : Chargement d'une séquence SemanticKITTI (depuis Google Drive).
3.  **Preprocessing** : Construction d'un nuage de points 4D (x, y, z, t) ou BEV-4D (x, y, t).
4.  **Clustering** : HGP-clusterer pour l'association spatio-temporelle avec choix de la fonction de splitting (Oracle ou Géométrique).
5.  **Evaluation** : Calcul de la métrique LSTQ (LiDAR Segmentation and Tracking Quality).
6.  **Visualisation** : Rendu 3D interactif.""", cell_type="markdown", title="r6k5rnJ01O2I")

# -----------------------------------------------------------------------------
# 2. Setup
# -----------------------------------------------------------------------------
add_cell("""# @title 1.1 Choix du Backend Géométrique
# 'geogram' est recommandé pour la vitesse (headless). 'cgal' est plus lent mais exact.
BACKEND = 'cgal'  # @param ['geogram', 'cgal']
print(f"Backend sélectionné : {BACKEND}")""", title="kN3IXi0p1O2L")

add_cell("""# @title 1.2 Installation des dépendances système
!apt-get update -qq
!apt-get install -y -qq build-essential cmake git libeigen3-dev libomp-dev

if BACKEND == 'cgal':
    # libboost-all-dev est souvent nécessaire pour que CMake détecte correctement CGAL
    !apt-get install -y -qq libcgal-dev libtbb-dev libtbbmalloc2 libgmp-dev libmpfr-dev libboost-all-dev
""", title="oXWQ4Fbd1O2M")

add_cell("""# @title 1.3 Installation des dépendances Python
!pip install -q --upgrade pip setuptools wheel Cython cmake jedi gdown pybind11
!pip install -q numpy scipy scikit-learn plotly tqdm joblib open3d plyfile hdbscan pandas matplotlib pyyaml""", title="wchS4VWb1O2N")

add_cell("""%%bash
# @title 1.4 Installation de HGP-clusterer et SemanticKITTI-API
set -euo pipefail
WORKDIR="/content"
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# HGP-clusterer
if [ -d HGP-clusterer ]; then
    git -C HGP-clusterer pull --ff-only
else
    git clone https://github.com/Ludwig-H/HGP-clusterer.git
fi

# SemanticKITTI API (pour l'évaluation)
if [ -d semantic-kitti-api ]; then
    git -C semantic-kitti-api pull --ff-only
else
    git clone https://github.com/PRBonn/semantic-kitti-api.git
fi""", title="X4Sn5xYc1O2O")

add_cell("""# @title 1.5 Compilation de HGP
import os
import sys
import subprocess

WORKDIR = "/content"
os.chdir(WORKDIR)

if BACKEND == 'geogram':
    if not os.path.exists('geogram'):
        print("Clonage de Geogram...")
        !git clone --recursive https://github.com/BrunoLevy/geogram.git
    
    print("Compilation de Geogram (Headless)...")
    !cmake -S geogram -B geogram/build -DCMAKE_BUILD_TYPE=Release -DGEOGRAM_WITH_GRAPHICS=OFF -DGEOGRAM_WITH_LUA=OFF -DGEOGRAM_WITH_GARGANTUA=OFF
    !cmake --build geogram/build --config Release --parallel 4
    !cmake --install geogram/build --prefix /usr/local
    os.environ['GEOGRAM_INSTALL_PREFIX'] = '/usr/local'

elif BACKEND == 'cgal':
    print("Configuration CGAL...")
    
    # -- FIX: Add CGAL path to environment for setup_cgal.py --
    cgal_prefix = "/usr/lib/x86_64-linux-gnu/cmake/CGAL"
    current_cpp = os.environ.get("CMAKE_PREFIX_PATH", "")
    os.environ["CMAKE_PREFIX_PATH"] = f"{current_cpp}:{cgal_prefix}" if current_cpp else cgal_prefix
    # ---------------------------------------------------------

    # On tente de construire l'outil CGAL, mais on continue même en cas d'erreur
    # car le setup.py principal pourrait réussir autrement.
    try:
        subprocess.run(["python3", f"{WORKDIR}/HGP-clusterer/scripts/setup_cgal.py"], check=True)
    except subprocess.CalledProcessError:
        print("⚠️ Attention: Echec du script setup_cgal.py. Tentative de continuation avec le build principal...")

os.chdir(f"{WORKDIR}/HGP-clusterer")
!rm -rf build dist *.egg-info

install_cmd = "pip install --no-build-isolation -v --no-deps ."
if BACKEND == 'geogram':
    install_cmd = f"GEOGRAM_INSTALL_PREFIX=/usr/local {install_cmd}"
elif BACKEND == 'cgal':
    # Ajout du chemin système pour CGAL (Debian/Ubuntu/Colab)
    # Note: On le passe aussi explicitement ici pour être sûr
    install_cmd = f"CGALDELAUNAY_ROOT={WORKDIR}/HGP-clusterer/CGALDelaunay CMAKE_PREFIX_PATH={WORKDIR}/HGP-clusterer:{cgal_prefix} {install_cmd}"

print(f"Exécution : {install_cmd}")
!{install_cmd}

os.environ["CGALDELAUNAY_ROOT"] = f"{WORKDIR}/HGP-clusterer/CGALDelaunay"

try:
    from hgp_clusterer import HGPClusterer
    print("✅ HGPClusterer installé.")
except ImportError as e:
    print(f"❌ Erreur import HGP: {e}")""", title="Wt1wHunY1O2P")

# -----------------------------------------------------------------------------
# 3. Data Loading
# -----------------------------------------------------------------------------
add_cell("""# @title 2.1 Configuration Séquence et Téléchargement
# IMPORTANT : Si vous ne voulez tester qu'une seule séquence, lancez cette cellule.
# Le téléchargement via gdown --folder récupère tout le dossier si on ne filtre pas.
# Ici, on télécharge tout le dataset SemanticKITTI (partiel) fourni via le lien Drive.

SEQUENCE_TO_TEST = 8 # @param {type:"integer"}
DOWNLOAD_DATA = True # @param {type:"boolean"}

# Choix du mode de téléchargement :
# - 'Folder' : Télécharge fichier par fichier (Très lent pour 10k fichiers, mais utile si on a que le lien du dossier)
# - 'Zip' : Télécharge une archive unique et décompresse (Beaucoup plus rapide, recommandé)
DOWNLOAD_MODE = "Folder" # @param ["Folder", "Zip"]

# IDs Google Drive par séquence
# Remplissez ce dictionnaire avec les IDs des dossiers ou des zips pour chaque séquence.
SEQUENCE_DRIVE_IDS = {
    8: {
        "Folder": "1UqFKvekjyic6L_8KD1kcv8MuGmQMIk0A",
        "Zip": "" # METTRE L'ID DU FICHIER ZIP ICI SI DISPONIBLE
    }
}

# Dossier Racine (Fallback si ID spécifique non trouvé en mode Folder)
ROOT_FOLDER_ID = "1ORVzSo-TWbNHeAC0-k3mxX9AiJHI_tVu"

if DOWNLOAD_DATA:
    import os
    import shutil
    
    # Destination racine
    base_dest = "/content/semantic_kitti_data"
    seq_str = f"{SEQUENCE_TO_TEST:02d}"
    target_dir = os.path.join(base_dest, seq_str)
    
    if not os.path.exists(target_dir):
        print(f"Démarrage du téléchargement (Mode : {DOWNLOAD_MODE})...")
        
        # Récupération des IDs pour la séquence choisie
        seq_ids = SEQUENCE_DRIVE_IDS.get(SEQUENCE_TO_TEST, {})
        
        if DOWNLOAD_MODE == "Zip":
            zip_id = seq_ids.get("Zip")
            if not zip_id:
                print(f"⚠️ Aucun ID Zip trouvé pour la séquence {SEQUENCE_TO_TEST}. Veuillez remplir SEQUENCE_DRIVE_IDS.")
                print("Passage automatique en mode Folder (Fallback)...")
                # On ne lance pas d'erreur, on essaie le folder si possible, sinon root
                DOWNLOAD_MODE = "Folder" 
            else:
                print(f"Téléchargement de l'archive Zip (ID: {zip_id})...")
                zip_path = os.path.join(base_dest, "sequence.zip")
                os.makedirs(base_dest, exist_ok=True)
                
                # Téléchargement
                !gdown {zip_id} -O {zip_path} --quiet
                
                print("Décompression...")
                # On décompresse
                !unzip -q {zip_path} -d {target_dir}
                !rm {zip_path}
                
                # Vérification de la structure (si le zip contenait un sous-dossier, on remonte)
                if not os.path.exists(os.path.join(target_dir, "velodyne")):
                    # Tentative de correction automatique
                    sub_dirs = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
                    if len(sub_dirs) == 1:
                        inner_dir = os.path.join(target_dir, sub_dirs[0])
                        print(f"Structure imbriquée détectée, déplacement de {inner_dir} vers {target_dir}...")
                        for item in os.listdir(inner_dir):
                            shutil.move(os.path.join(inner_dir, item), target_dir)
                        os.rmdir(inner_dir)

        # Note: Ce bloc est exécuté si mode Folder OU si fallback depuis Zip
        if DOWNLOAD_MODE == "Folder":
            folder_id = seq_ids.get("Folder")
            if not folder_id:
                # Fallback sur le root folder (pas idéal mais fonctionnel)
                print(f"ID spécifique Folder manquant pour la séquence {SEQUENCE_TO_TEST}.")
                print(f"Tentative de téléchargement via le dossier racine {ROOT_FOLDER_ID}...")
                !gdown --folder {ROOT_FOLDER_ID} -O {base_dest} --quiet --remaining-ok
            else:
                print(f"Téléchargement du dossier (ID: {folder_id})...")
                # On crée le dossier de la séquence pour gdown
                !gdown --folder {folder_id} -O {target_dir} --quiet --remaining-ok
        
        print("Téléchargement terminé.")
    else:
        print(f"Dossier {target_dir} existe déjà. Skip download.")
else:
    print("Téléchargement désactivé.")

print(f"Séquence cible pour le test : {SEQUENCE_TO_TEST}")""", title="A9Y4dn0P1O2P")

add_cell("""# @title 2.2 Loader SemanticKITTI
import os
import numpy as np
import glob

class SemanticKITTILoader:
    def __init__(self, base_path, sequence_num):
        self.seq_str = f"{sequence_num:02d}"

        # Recherche du dossier de la séquence.
        # Structure attendue : base_path/08 ou base_path/sequences/08

        # 1. Chercher direct
        possible_paths = glob.glob(f"{base_path}/{self.seq_str}")

        # 2. Chercher dans un sous-dossier 'sequences' (structure officielle KITTI)
        if not possible_paths:
            possible_paths = glob.glob(f"{base_path}/**/sequences/{self.seq_str}", recursive=True)

        # 3. Chercher récursivement n'importe où (au cas où gdown a créé une structure intermédiaire)
        if not possible_paths:
             possible_paths = glob.glob(f"{base_path}/**/{self.seq_str}", recursive=True)

        # Filtrer pour ne garder que les vrais dossiers contenant 'velodyne'
        valid_paths = []
        for p in possible_paths:
            if os.path.exists(os.path.join(p, 'velodyne')):
                valid_paths.append(p)

        if not valid_paths:
            raise ValueError(f"Séquence {self.seq_str} introuvable dans {base_path}. Vérifiez que le dossier 'velodyne' est bien présent.")

        self.seq_path = valid_paths[0]
        print(f"Séquence chargée : {self.seq_path}")

        self.velo_path = os.path.join(self.seq_path, 'velodyne')
        self.label_path = os.path.join(self.seq_path, 'labels')
        self.poses_file = os.path.join(self.seq_path, 'poses.txt')
        self.calib_file = os.path.join(self.seq_path, 'calib.txt')

        # Fallback pour poses.txt/calib.txt s'ils sont dans le dossier parent (structure dataset/sequences/08)
        if not os.path.exists(self.poses_file):
             # Essayer de remonter d'un niveau (dataset/sequences/) ou deux
             parent = os.path.dirname(self.seq_path) # dataset/sequences
             grandparent = os.path.dirname(parent) # dataset

             # Cas dataset/poses.txt (peu probable mais...)
             # Cas dataset/sequences/08/poses.txt (standard)
             pass

        self.scan_files = sorted(glob.glob(os.path.join(self.velo_path, '*.bin')))
        self.label_files = sorted(glob.glob(os.path.join(self.label_path, '*.label')))
        self.poses = self._load_poses()
        self.calib = self._load_calib()

    def _load_poses(self):
        if not os.path.exists(self.poses_file):
            print(f"Info: poses.txt non trouvé ({self.poses_file}).")
            return []
        poses = []
        with open(self.poses_file, 'r') as f:
            for line in f:
                values = [float(v) for v in line.strip().split()]
                pose = np.vstack([np.array(values).reshape(3, 4), [0, 0, 0, 1]])
                poses.append(pose)
        return poses

    def _load_calib(self):
        if not os.path.exists(self.calib_file):
            print(f"Info: calib.txt non trouvé ({self.calib_file}).")
            return np.eye(4)
        calib = {}
        with open(self.calib_file, 'r') as f:
            for line in f:
                if ':' not in line: continue
                key, val = line.split(':', 1)
                calib[key] = np.array([float(x) for x in val.split()]).reshape(3, 4)
        if 'Tr' in calib:
            return np.vstack([calib['Tr'], [0, 0, 0, 1]])
        return np.eye(4)

    def get_scan(self, idx, apply_pose=True):
        scan = np.fromfile(self.scan_files[idx], dtype=np.float32).reshape(-1, 4)
        points = scan[:, :3]
        if apply_pose and self.poses and idx < len(self.poses):
            T = self.poses[idx] @ self.calib
            points = (T @ np.hstack([points, np.ones((len(points), 1))]).T).T[:, :3]
        return points

    def get_labels(self, idx):
        if idx >= len(self.label_files): return None, None
        label = np.fromfile(self.label_files[idx], dtype=np.uint32)
        return label & 0xFFFF, label >> 16

    def __len__(self): return len(self.scan_files)""", title="mJ8rAu0N1O2Q")

# -----------------------------------------------------------------------------
# 4. 4D Construction
# -----------------------------------------------------------------------------
add_cell("""# @title 3.1 Construction du Nuage 4D
import numpy as np

START_FRAME = 0 # @param {type:"integer"}
NUM_FRAMES = 10 # @param {type:"integer"}
DT_SCALE = 0.5  # @param {type:"number"}
APPLY_BEV = True # @param {type:"boolean"}
# Mode Sémantique :
# - 'Oracle' : Utilise la vérité terrain fournie par SemanticKITTI pour filtrer les objets mobiles (Things).
# - 'None' : Ne filtre rien (Lance le clustering sur absolument toute la scène, lent et non recommandé).
# (Note: Le chargement d'une prédiction réseau externe viendrait ici dans une future mise à jour)
SEMANTIC_MODE = "Oracle" # @param ["Oracle", "None"]

# SemanticKITTI classes "things" (véhicules, piétons, cyclistes...)
# Classes : 10, 11, 13, 15, 16, 18, 20, 30, 31, 32 et leurs équivalents "moving" (>250)
THINGS_CLASSES = set([10, 11, 13, 15, 16, 18, 20, 30, 31, 32, 252, 253, 254, 255, 256, 257, 258, 259])

# Initialisation des variables pour éviter les NameError
X_clustering = None
X_4d = None
Y_sem = None
Y_inst = None
Time_idx = None
Original_Indices = None # Pour garder la trace si on filtre

try:
    loader = SemanticKITTILoader("/content/semantic_kitti_data", SEQUENCE_TO_TEST)
    points_4d, gt_sem, gt_inst, times, indices = [], [], [], [], []
    
    total_points = 0
    print(f"Chargement frames {START_FRAME} -> {START_FRAME + NUM_FRAMES}...")
    for i in range(NUM_FRAMES):
        idx = START_FRAME + i
        if idx >= len(loader): break

        pts = loader.get_scan(idx, apply_pose=True)
        s, inst = loader.get_labels(idx)

        # 4D Point: x, y, z, t
        t_col = np.full((len(pts), 1), i * DT_SCALE)
        pts_4d = np.hstack([pts, t_col])
        
        # Filtre Sémantique
        if SEMANTIC_MODE == "Oracle":
            # On ne garde que les classes "Things"
            mask = np.array([sem in THINGS_CLASSES for sem in s])
            pts_4d = pts_4d[mask]
            s = s[mask]
            inst = inst[mask]
            
            # Si on veut garder l'index original par rapport à la frame (pour de la visulaisation par ex)
            frame_indices = np.arange(len(mask))[mask]
        else:
            frame_indices = np.arange(len(pts))

        points_4d.append(pts_4d)
        gt_sem.append(s)
        gt_inst.append(inst)
        times.extend([i] * len(pts_4d))
        indices.append(frame_indices + total_points)
        total_points += len(pts) # On ajoute le total brut pour les indices absolus

    if points_4d:
        X_4d = np.vstack(points_4d)
        Y_sem = np.hstack(gt_sem)
        Y_inst = np.hstack(gt_inst)
        Time_idx = np.array(times)
        Original_Indices = np.hstack(indices)

        # Bird's Eye View : on utilise uniquement x, y, t pour le clustering
        if APPLY_BEV:
            print(f"Mode Bird's-Eye-View (BEV) activé : Clustering sur (x, y, t).")
            X_clustering = np.column_stack([X_4d[:, 0], X_4d[:, 1], X_4d[:, 3]])
        else:
            print("Mode 4D Complet activé : Clustering sur (x, y, z, t).")
            X_clustering = X_4d

        print(f"Sémantique : Mode {SEMANTIC_MODE}.")
        print(f"Nuage 4D filtré: {X_4d.shape} points conservés.")
        print(f"Input Clustering: {X_clustering.shape}")
    else:
        print("Aucun point chargé ou aucun point 'thing' trouvé. Vérifiez les chemins.")

except Exception as e:
    print(f"Erreur lors du chargement des données: {e}")
    print("---------------------------------------------------------")
    print("⚠️ GÉNÉRATION DE DONNÉES SYNTHÉTIQUES (FALLBACK) ⚠️")
    print("---------------------------------------------------------")
    from sklearn.datasets import make_blobs
    n_samples = 5000
    X_syn, y_syn = make_blobs(n_samples=n_samples, n_features=3, centers=5, cluster_std=1.0)
    t_syn = np.random.randint(0, NUM_FRAMES, size=n_samples) * DT_SCALE
    X_4d = np.column_stack([X_syn, t_syn])
    Y_sem = np.zeros(n_samples, dtype=int)
    Y_inst = y_syn + 1 
    Time_idx = (t_syn / DT_SCALE).astype(int)
    X_clustering = X_4d if not APPLY_BEV else X_4d[:, [0, 1, 3]]
    SEMANTIC_MODE = "None"
    print(f"Données synthétiques générées: {X_clustering.shape}")""", title="-7bzecQb1O2Q")

# -----------------------------------------------------------------------------
# 5. Clustering
# -----------------------------------------------------------------------------
add_cell("""# @title 4.1 HGP Clustering
import time
import numpy as np

# --- Import sécurisé de HGPClusterer ---
try:
    from hgp_clusterer import HGPClusterer
except ImportError:
    print("⚠️ Module HGPClusterer introuvable. Tentative de correction du path...")
    import sys
    if "/content/HGP-clusterer/src" not in sys.path:
        sys.path.append("/content/HGP-clusterer/src")
    try:
        from hgp_clusterer import HGPClusterer
        print("✅ HGPClusterer importé avec succès après correction du path.")
    except ImportError as e:
        raise RuntimeError(f"❌ Impossible d'importer HGPClusterer même après correction. Erreur: {e}. Veuillez vérifier la compilation en section 1.5.")

K = 3 # @param {type:"integer"}
MIN_CLUSTER_SIZE = 50 # @param {type:"integer"}
SPLIT_MODE = "None" # @param ["None", "Oracle", "Geometric"]

# Vérification préalable des données
if X_clustering is None or len(X_clustering) == 0:
    raise RuntimeError("Erreur critique: X_clustering est vide ou n'est pas défini. Veuillez vérifier la cellule 'Construction du Nuage 4D'.")

# Initialisation du tableau global des prédictions (ID des instances)
# -1 indique le bruit/non-assigné
labels_pred = np.full(len(X_clustering), -1, dtype=int)
global_instance_offset = 0

# --- Boucle par classe sémantique ---
# Pour une segmentation panoptique propre, le clustering est indépendant pour chaque classe "Thing".

unique_classes = np.unique(Y_sem)
print(f"Classes sémantiques trouvées dans ce bloc de frames : {unique_classes}")

for semantic_class in unique_classes:
    if semantic_class not in THINGS_CLASSES and SEMANTIC_MODE == "Oracle":
        # Ce cas ne devrait pas arriver si le nuage a été pré-filtré, mais par sécurité :
        continue
    
    # Masque pour extraire uniquement les points de la classe courante
    class_mask = (Y_sem == semantic_class)
    X_class = X_clustering[class_mask]
    Y_inst_class = Y_inst[class_mask] # Utile si Oracle Split

    if len(X_class) < MIN_CLUSTER_SIZE:
        # Pas assez de points pour former un cluster, on les laisse à -1
        print(f"Classe {semantic_class} ignorée (seulement {len(X_class)} points, min={MIN_CLUSTER_SIZE}).")
        continue

    print(f"--- Traitement de la classe sémantique {semantic_class} ({len(X_class)} points) ---")

    # --- Fonctions de Splitting adaptées au sous-masque ---
    def oracle_split(parent, children):
        "Split si les enfants sont nettement plus purs sémantiquement que le parent."
        p_labels = Y_inst_class[parent]; p_labels = p_labels[p_labels > 0]
        if len(p_labels) == 0: return False

        parent_purity = np.max(np.unique(p_labels, return_counts=True)[1]) / len(p_labels)

        child_purity_sum = 0; total = 0
        for c in children:
            c_labels = Y_inst_class[c]; c_labels = c_labels[c_labels > 0]
            if len(c_labels) > 0:
                child_purity_sum += np.max(np.unique(c_labels, return_counts=True)[1])
                total += len(c_labels)

        child_purity = child_purity_sum / total if total > 0 else 0
        return child_purity > parent_purity + 0.05

    def geometric_split(parent, children):
        "Split si le parent est géométriquement incohérent (variance trop élevée) comparé aux enfants."
        pts_parent = X_class[parent]
        var_parent = np.var(pts_parent[:, :2], axis=0).sum()

        weighted_var_children = 0; total_len = 0
        for c in children:
            pts_child = X_class[c]
            if len(pts_child) < 2: continue
            v = np.var(pts_child[:, :2], axis=0).sum()
            weighted_var_children += v * len(pts_child)
            total_len += len(pts_child)

        avg_var_children = weighted_var_children / total_len if total_len > 0 else var_parent
        if var_parent > 1.5 * avg_var_children and var_parent > 5.0:
            return True
        return False

    # --- Sélection de la fonction ---
    if SPLIT_MODE == "Oracle":
        split_func = oracle_split
    elif SPLIT_MODE == "Geometric":
        split_func = geometric_split
    else:
        split_func = None

    clusterer = HGPClusterer(
        K=K, min_cluster_size=MIN_CLUSTER_SIZE, min_samples=K+1,
        splitting=split_func,
        backend=BACKEND,
        cgal_root=os.environ.get("CGALDELAUNAY_ROOT"),
        verbose=False # Rendu silencieux pour ne pas spammer la boucle
    )

    t0 = time.time()
    try:
        class_labels_pred = clusterer.fit_predict(X_class)
        num_clusters = len(set(class_labels_pred)) - (1 if -1 in class_labels_pred else 0)
        print(f"  -> {num_clusters} clusters trouvés en {time.time()-t0:.2f}s.")
        
        # On assigne des IDs globaux uniques à ces nouveaux clusters
        valid_cluster_mask = class_labels_pred >= 0
        if np.any(valid_cluster_mask):
            # Décalage des labels pour qu'ils soient uniques sur toute la scène
            class_labels_pred[valid_cluster_mask] += global_instance_offset
            
            # Injection dans le tableau global
            labels_pred[class_mask] = class_labels_pred
            
            # Mise à jour de l'offset pour la prochaine classe
            global_instance_offset = np.max(class_labels_pred) + 1
            
    except Exception as e:
        print(f"  -> Erreur durant le clustering de la classe {semantic_class}: {e}")

print(f"\\nClustering global terminé. Total instances uniques détectées : {global_instance_offset}")""", title="tT_pybI-1O2R")

# -----------------------------------------------------------------------------
# 6. Evaluation (LSTQ)
# -----------------------------------------------------------------------------
add_cell("""# @title 5.1 Évaluation LSTQ (Panoptic)
# Cette cellule calcule une approximation du LSTQ (LiDAR Segmentation and Tracking Quality)
# S_assoc : Association Quality (Tracking IoU)
# S_cls   : Classification Quality (Semantic IoU - ici on suppose la sémantique connue/oracle pour l'instant)

def compute_lstq_simplified(pred_labels, gt_inst_labels, gt_sem_labels):
    \"\"\"
    Calcul simplifié de S_assoc.
    On associe chaque cluster prédit à l'instance GT majoritaire.
    \"\"\"
    # Ignorer le bruit
    mask = (gt_inst_labels > 0) & (pred_labels >= 0)
    if np.sum(mask) == 0: return 0.0, 0.0

    # Intersection Matrix: rows=GT, cols=Pred
    # C'est une approximation, le vrai LSTQ utilise une association hongroise sur les tubes 4D

    from sklearn.metrics import confusion_matrix

    # Remap labels to 0..N for confusion matrix
    u_gt, inv_gt = np.unique(gt_inst_labels[mask], return_inverse=True)
    u_pred, inv_pred = np.unique(pred_labels[mask], return_inverse=True)

    cm = confusion_matrix(inv_gt, inv_pred)

    # T-IoU pour chaque paire (GT_i, Pred_j)
    # IoU = Intersection / Union
    # Intersection = cm[i, j]
    # Union = count_gt[i] + count_pred[j] - cm[i, j]

    count_gt = cm.sum(axis=1)
    count_pred = cm.sum(axis=0)

    # Association greedy : Pour chaque GT, on prend le Pred qui maximise l'IoU
    ious = []
    for i in range(len(u_gt)):
        best_iou = 0
        for j in range(len(u_pred)):
            inter = cm[i, j]
            union = count_gt[i] + count_pred[j] - inter
            if union > 0:
                iou = inter / union
                if iou > best_iou: best_iou = iou
        ious.append(best_iou)

    s_assoc = np.mean(ious) if ious else 0.0
    return s_assoc

if X_clustering is not None and len(X_clustering) > 0:
    print("Calcul du LSTQ (Approximation S_assoc)...")
    s_assoc = compute_lstq_simplified(labels_pred, Y_inst, Y_sem)
    print(f"S_assoc (Tracking Quality) : {s_assoc:.4f}")
    print("Note: Ceci est une approximation. Pour le score officiel, utilisez evaluate_panoptic.py de l'API SemanticKITTI.")
else:
    print("Pas de données pour l'évaluation.")""", title="eQImb5yp1O2R")

# -----------------------------------------------------------------------------
# 7. Visualization
# -----------------------------------------------------------------------------
add_cell("""# @title 6.1 Visualisation 3D
import plotly.graph_objects as go

if X_4d is not None and len(X_4d) > 0:
    idx = np.arange(0, len(X_4d), 10) # Downsample
    X_v = X_4d[idx]
    L_v = labels_pred[idx]

    fig = go.Figure()
    u_l = np.unique(L_v)

    for l in u_l:
        if l == -1: continue
        m = L_v == l
        fig.add_trace(go.Scatter3d(
            x=X_v[m, 0], y=X_v[m, 1], z=X_v[m, 2],
            mode='markers', marker=dict(size=2), name=f'C{l}'
        ))
    fig.update_layout(title="HGP Clusters 4D", scene=dict(aspectmode='data'))
    fig.show()
else:
    print("Pas de données à afficher.")""", title="d0t8xdfp1O2S")

# Save notebook
with open('tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)
