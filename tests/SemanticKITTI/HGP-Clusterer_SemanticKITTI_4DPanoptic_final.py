try:
    from google.colab import output
    output.enable_custom_widget_manager()
except ImportError:
    pass

import sys
sys.setrecursionlimit(10000)

# @title 0. Montage Google Drive (Si exécution sur Colab)
import os
try:
    from google.colab import drive
    os.makedirs('/content/drive', exist_ok=True)
    drive.mount('/content/drive')
    print("Google Drive monté avec succès !")
except ImportError:
    print("Environnement hors Colab détecté. Montage Drive ignoré.")


try:
    from google.colab import output
    output.enable_custom_widget_manager()
except ImportError:
    pass
# @title 1.1 Choix du Backend Géométrique
# 'geogram' est recommandé pour la vitesse (headless). 'cgal' est plus lent mais exact.
BACKEND = 'geogram'  # @param ['geogram', 'cgal']
print(f"Backend sélectionné : {BACKEND}")

# @title 1.2 Installation des dépendances système
# os.system('apt-get update -qq')
# os.system('apt-get install -y -qq build-essential cmake git libeigen3-dev libomp-dev')

if BACKEND == 'cgal':
    # libboost-all-dev est souvent nécessaire pour que CMake détecte correctement CGAL
# os.system('apt-get install -y -qq libcgal-dev libtbb-dev libtbbmalloc2 libgmp-dev libmpfr-dev libboost-all-dev')

# @title 1.3 Installation des dépendances Python
# os.system('pip install -q --upgrade pip setuptools wheel Cython cmake jedi gdown pybind11')
# os.system('pip install -q numpy scipy scikit-learn plotly tqdm joblib open3d plyfile hdbscan pandas matplotlib pyyaml shapely mip POT')


%%bash
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
fi

try:
    from google.colab import output
    output.enable_custom_widget_manager()
except ImportError:
    pass
# @title 1.5 Compilation de HGP
import os
import sys
import subprocess

WORKDIR = "/content"
os.chdir(WORKDIR)

if BACKEND == 'geogram':
    if not os.path.exists('geogram'):
        print("Clonage de Geogram...")
# os.system('git clone --recursive https://github.com/BrunoLevy/geogram.git')
    
    print("Compilation de Geogram (Headless)...")
    os.system('rm -rf geogram/build')
    os.system('cmake -S geogram -B geogram/build -DCMAKE_BUILD_TYPE=Release -DGEOGRAM_WITH_GRAPHICS=OFF -DGEOGRAM_WITH_LUA=OFF -DGEOGRAM_WITH_GARGANTUA=OFF')
# os.system('cmake --build geogram/build --config Release --parallel 4')
# os.system('cmake --install geogram/build --prefix /usr/local')
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
# os.system('rm -rf build dist *.egg-info')

install_cmd = "pip install --no-build-isolation -v --no-deps ."
if BACKEND == 'geogram':
    install_cmd = f"GEOGRAM_INSTALL_PREFIX=/usr/local {install_cmd}"
elif BACKEND == 'cgal':
    # Ajout du chemin système pour CGAL (Debian/Ubuntu/Colab)
    # Note: On le passe aussi explicitement ici pour être sûr
    install_cmd = f"CGALDELAUNAY_ROOT={WORKDIR}/HGP-clusterer/CGALDelaunay CMAKE_PREFIX_PATH={WORKDIR}/HGP-clusterer:{cgal_prefix} {install_cmd}"

print(f"Exécution : {install_cmd}")
# os.system('{install_cmd}')

os.environ["CGALDELAUNAY_ROOT"] = f"{WORKDIR}/HGP-clusterer/CGALDelaunay"

try:
    from hgp_clusterer import HGPClusterer
    print("✅ HGPClusterer installé.")
except ImportError as e:
    print(f"❌ Erreur import HGP: {e}")

# @title 2.1 Configuration Séquence et Téléchargement
# IMPORTANT : Si vous ne voulez tester qu'une seule séquence, lancez cette cellule.
# Le téléchargement via gdown --folder récupère tout le dossier si on ne filtre pas.
# Ici, on télécharge tout le dataset SemanticKITTI (partiel) fourni via le lien Drive.

SEQUENCE_TO_TEST = 8 # @param {type:"integer"}
DOWNLOAD_DATA = True # @param {type:"boolean"}

START_FRAME = 0 # @param {type:"integer"}
NUM_FRAMES = 200 # @param {type:"integer"} (-1 pour toutes les frames)
DT_SCALE = 0.1  # @param {type:"number"}
APPLY_BEV = False # @param {type:"boolean"}
# Mode Sémantique :
# - 'Oracle' : Utilise la vérité terrain fournie par SemanticKITTI pour filtrer les objets mobiles (Things).
# - 'None' : Ne filtre rien (Lance le clustering sur absolument toute la scène, lent et non recommandé).
# (Note: Le chargement d'une prédiction réseau externe viendrait ici dans une future mise à jour)
SEMANTIC_MODE = "Oracle" # @param ["Oracle", "None"]

# Choix du mode de téléchargement :
# - 'Folder' : Télécharge fichier par fichier (Très lent pour 10k fichiers, mais utile si on a que le lien du dossier)
# - 'Zip' : Télécharge une archive unique et décompresse (Beaucoup plus rapide, recommandé)
DOWNLOAD_MODE = "Zip" # @param ["Folder", "Zip"]

# IDs Google Drive par séquence
# Remplissez ce dictionnaire avec les IDs des dossiers ou des zips pour chaque séquence.
SEQUENCE_DRIVE_IDS = {
    8: {
        "Folder": "1UqFKvekjyic6L_8KD1kcv8MuGmQMIk0A",
        "Zip": "1ZoZtzdFAkPWYHT8sFsHjwyEmFHbpaIQH"
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
                print(f"Préparation de l'archive Zip pour la séquence {seq_str}...")
                zip_path = os.path.join(base_dest, "sequence.zip")
                os.makedirs(base_dest, exist_ok=True)
                
                local_zip = f"/content/drive/MyDrive/Datasets/semantic_kitti/sequences_zip/{seq_str}.zip"
                success_local = False
                
                # Tentative 1: Drive local prioritaire
                if os.path.exists(local_zip):
                    print(f"Fichier trouvé sur Drive ({local_zip}), copie en cours...")
                    shutil.copy(local_zip, zip_path)
                    success_local = True
                else:
                    # Tentative de montage si on est sur Colab
                    try:
                        from google.colab import drive
                        if not os.path.exists('/content/drive/MyDrive'):
                            print("Montage de Google Drive...")
                            os.makedirs('/content/drive', exist_ok=True)
                            drive.mount('/content/drive')
                        if os.path.exists(local_zip):
                            print(f"Fichier trouvé sur Drive ({local_zip}), copie en cours...")
                            shutil.copy(local_zip, zip_path)
                            success_local = True
                    except ImportError:
                        pass
                
                # Tentative 2: Téléchargement public via Gdown (Fallback)
                if not success_local or not os.path.exists(zip_path):
                    print(f"Archive locale introuvable. Téléchargement via Gdown (ID: {zip_id})...")
                    res = os.system(f"gdown {zip_id} -O {zip_path} --quiet")
                    if res != 0 or not os.path.exists(zip_path):
                        print("Erreur Gdown. L'archive n'a pas pu être récupérée.")
                
                if os.path.exists(zip_path):
                    print("Décompression...")
                    # On décompresse
                    os.system(f"unzip -q {zip_path} -d {target_dir}")
                    os.system(f"rm -f {zip_path}")
                    
                    # Vérification de la structure (si le zip contenait un sous-dossier, on remonte)
                    if os.path.exists(target_dir) and not os.path.exists(os.path.join(target_dir, "velodyne")):
                        # Tentative de correction automatique
                        sub_dirs = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
                        if len(sub_dirs) >= 1:
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
                folder_id = ROOT_FOLDER_ID
                dl_target = base_dest
            else:
                dl_target = target_dir

            print(f"Téléchargement du dossier (ID: {folder_id})...")
            # --- TÉLÉCHARGEMENT SÉLECTIF ---
            try:
                import gdown
                print("Analyse du contenu du Drive... (limité aux 50 premiers fichiers par dossier par Google Drive)")
                # remaining_ok=True permet de récupérer les 50 premiers fichiers sans crasher
                files = gdown.download_folder(id=folder_id, skip_download=True, quiet=True, remaining_ok=True)
                if files:
                    needed_frames = set()
                    if NUM_FRAMES != -1:
                        needed_frames = set(range(START_FRAME, START_FRAME + NUM_FRAMES))
                    
                    downloaded_count = 0
                    
                    for f in files:
                        try:
                            f_path = f.path if hasattr(f, 'path') else f.get('path', '')
                            f_id = f.id if hasattr(f, 'id') else f.get('id', '')
                        except:
                            continue
                            
                        # Vérifier si c'est un fichier lié à une frame
                        filename = f_path.split('/')[-1] if '/' in f_path else f_path
                        is_frame_file = False
                        frame_idx = -1
                        
                        if filename.endswith('.bin') or filename.endswith('.label'):
                            try:
                                frame_idx = int(filename.split('.')[0])
                                is_frame_file = True
                            except ValueError:
                                pass
                                
                        # Filtrer
                        if is_frame_file and NUM_FRAMES != -1:
                            if frame_idx not in needed_frames:
                                continue # On skip cette frame
                        
                        # Créer le chemin local
                        import os
                        local_path = os.path.join(dl_target, f_path)
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                        
                        if not os.path.exists(local_path):
                            print(f"Téléchargement : {f_path}")
                            gdown.download(id=f_id, output=local_path, quiet=True)
                        downloaded_count += 1
                        
                    print(f"Téléchargement sélectif terminé ({downloaded_count} fichiers traités).")
                    
                    # Vérification si on a pu tout récupérer ou si on a tapé la limite des 50
                    # On s'attend à 2 fichiers par frame (bin + label) s'il s'agit de frames
                    # Mais s'il y a des fichiers autres (calib), on ne s'inquiète que si downloaded_count est faible.
                    # Pour être simple, on informe l'utilisateur si la frame demandée max n'est pas dans les 50.
                    if NUM_FRAMES != -1 and (START_FRAME + NUM_FRAMES) > 50:
                        print(f"\n⚠️ ATTENTION : Vous avez demandé jusqu'à la frame {START_FRAME + NUM_FRAMES - 1}.")
                        print("Google Drive limite le listage anonyme aux 50 premiers fichiers de chaque dossier (jusqu'à la frame 49).")
                        print("👉 Pour traiter au-delà de la frame 49, VEUILLEZ UTILISER LE MODE 'Zip' dans les paramètres.\n")
                else:
                    print("Impossible d'analyser le dossier. Veuillez utiliser le mode Zip.")
            except Exception as e:
                print(f"\n❌ Le téléchargement sélectif a échoué ({e}).")
                print("👉 VEUILLEZ UTILISER LE MODE 'Zip' DANS LES PARAMÈTRES CI-DESSUS.\n")
        print("Téléchargement terminé.")
    else:
        print(f"Dossier {target_dir} existe déjà. Skip download.")
else:
    print("Téléchargement désactivé.")

print(f"Séquence cible pour le test : {SEQUENCE_TO_TEST}")

try:
    from google.colab import output
    output.enable_custom_widget_manager()
except ImportError:
    pass
# @title 2.2 Loader SemanticKITTI
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
                calib[key.strip()] = np.array([float(x) for x in val.split()]).reshape(3, 4)
        if 'Tr' in calib:
            return np.vstack([calib['Tr'], [0, 0, 0, 1]])
        elif 'Tr_velo_to_cam' in calib:
            return np.vstack([calib['Tr_velo_to_cam'], [0, 0, 0, 1]])
        for k in calib.keys():
            if 'Tr' in k:
                return np.vstack([calib[k], [0, 0, 0, 1]])
        print("⚠️ Warning: Aucune matrice 'Tr' (Transformation Velo -> Cam) trouvée dans calib.txt. Fallback sur une matrice Identité (ce qui causera des erreurs de rotation sur l'axe Z !)")
        return np.eye(4)

    def get_pose(self, idx, flatten=True):
        if self.poses and idx < len(self.poses):
            Tr = self.calib
            Pose_cam = self.poses[idx]
            T_world_velo = np.linalg.inv(Tr) @ Pose_cam @ Tr
            if flatten:
                t = T_world_velo[:3, 3]
                R = T_world_velo[:3, :3]
                yaw = np.arctan2(R[1, 0], R[0, 0])
                R_flat = np.array([
                    [np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw),  np.cos(yaw), 0],
                    [0,           0,            1]
                ])
                T_world_velo = np.eye(4)
                T_world_velo[:3, :3] = R_flat
                T_world_velo[:3, 3] = t
            return T_world_velo
        return np.eye(4)

    def get_scan(self, idx, apply_pose=True, flatten_pose=True):
        scan = np.fromfile(self.scan_files[idx], dtype=np.float32).reshape(-1, 4)
        points = scan[:, :3]
        if apply_pose and self.poses and idx < len(self.poses):
            T_world_velo = self.get_pose(idx, flatten=flatten_pose)
            hom_points = np.hstack([points, np.ones((len(points), 1))])
            points = (T_world_velo @ hom_points.T).T[:, :3]
        return points
    def get_labels(self, idx):
        if idx >= len(self.label_files): return None, None
        label = np.fromfile(self.label_files[idx], dtype=np.uint32)
        return label & 0xFFFF, label >> 16

    def __len__(self): return len(self.scan_files)

# @title 3.1 Construction du Nuage 4D
import numpy as np


# Mapping officiel SemanticKITTI (fusionne les classes "moving" avec leur équivalent statique)
LEARNING_MAP = {
  0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5, 30: 6, 
  31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13, 51: 14, 52: 0, 
  60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19, 99: 0, 252: 1, 
  253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5
}
# Vecteur de mapping rapide (taille max 260)
max_key = max(LEARNING_MAP.keys())
LABEL_MAP_ARRAY = np.zeros(max_key + 1, dtype=np.uint32)
for k, v in LEARNING_MAP.items():
    LABEL_MAP_ARRAY[k] = v

# SemanticKITTI classes "things" dans l'espace mappé (1 à 8)
# 1:car, 2:bicycle, 3:motorcycle, 4:truck, 5:other-vehicle, 6:person, 7:bicyclist, 8:motorcyclist
THINGS_CLASSES = set([1, 2, 3, 4, 5, 6, 7, 8])

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
    actual_num_frames = len(loader) - START_FRAME if NUM_FRAMES == -1 else NUM_FRAMES
    print(f"Chargement frames {START_FRAME} -> {START_FRAME + actual_num_frames}...")
    for i in range(actual_num_frames):
        idx = START_FRAME + i
        if idx >= len(loader): break

        pts = loader.get_scan(idx, apply_pose=True)
        s_raw, inst = loader.get_labels(idx)
        
        # Mapping des labels sémantiques bruts vers l'espace d'évaluation
        s = LABEL_MAP_ARRAY[s_raw]

        # 4D Point: x, y, z, t
        t_col = np.full((len(pts), 1), i * DT_SCALE)
        pts_4d = np.hstack([pts[:, 0:3], t_col]) # Colonnes: 0:x, 1:y, 2:z, 3:t
        
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
            print(f"Mode Bird's-Eye-View (BEV) activé : Clustering sur (x, y) [Streaming 2D BEV].")
            X_clustering = np.column_stack([X_4d[:, 0], X_4d[:, 1]])
        else:
            print("Mode 4D Complet activé : Clustering sur (x, y, z) [Streaming 3D].")
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
    synth_frames = NUM_FRAMES if NUM_FRAMES != -1 else 10
    t_syn = np.random.randint(0, synth_frames, size=n_samples) * DT_SCALE
    X_4d = np.column_stack([X_syn, t_syn])
    Y_sem = np.zeros(n_samples, dtype=int)
    Y_inst = y_syn + 1 
    Time_idx = (t_syn / DT_SCALE).astype(int)
    X_clustering = X_4d[:, :3] if not APPLY_BEV else X_4d[:, [0, 1]]
    print(f"Données synthétiques générées: {X_clustering.shape}")

try:
    from google.colab import output
    output.enable_custom_widget_manager()
except ImportError:
    pass
# @title 4.1 HGP Clustering & Optimal MILP 4D Panoptic Tracking
import time
import numpy as np
import os
import torch
import sys
import itertools
from collections import defaultdict
from shapely.geometry import MultiPoint, GeometryCollection
from shapely.strtree import STRtree
import mip

# Import sécurisé HGP
if "/content/HGP-clusterer/src" in sys.path:
    sys.path.remove("/content/HGP-clusterer/src")
if "" in sys.path:
    sys.path.remove("")
try:
    from hgp_clusterer import HGPClusterer
except ImportError as e:
    print(f"❌ Erreur import HGP: {e}")
    raise

# --- Paramètres ---
K = 3 # @param {type:"integer"}
MIN_CLUSTER_SIZE = 10 # @param {type:"integer"}
DBSCAN_FACTOR = 0.8 # @param {type:"number"}
EXP_Z = 1 # @param {type:"number"}
TRACKING_VERBOSE = True # @param {type:"boolean"}
HGP_VERBOSE = False # @param {type:"boolean"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# 1. OUTILS TOPOLOGIQUES ET GÉOMÉTRIQUES O(1)
# ==========================================================
def get_global_hull(nodes, node_hulls):
    geoms = [node_hulls[v] for v in nodes if v in node_hulls]
    if not geoms: return None
    hull = GeometryCollection(geoms).convex_hull
    if hull.geom_type in ['Point', 'LineString']: 
        hull = hull.buffer(1e-5)
    return hull

def hulls_intersect(nodes1, nodes2, node_hulls):
    h1 = get_global_hull(nodes1, node_hulls)
    h2 = get_global_hull(nodes2, node_hulls)
    return h1.intersects(h2) if (h1 and h2) else False

def get_minimal_witness(nodes1, nodes2, node_hulls):
    w1, w2 = list(nodes1), list(nodes2)
    for v in list(w1):
        if len(w1) == 1: continue
        w1.remove(v)
        if not hulls_intersect(w1, w2, node_hulls): w1.append(v)
    for u in list(w2):
        if len(w2) == 1: continue
        w2.remove(u)
        if not hulls_intersect(w1, w2, node_hulls): w2.append(u)
    return w1, w2

# ==========================================================
# 2. LE SOLVEUR MILP EXACT

def compute_obb_volume(pts, is_bev=False):
    if len(pts) < 3:
        return 0.0
    xy = pts[:, :2]
    cov = np.cov(xy.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    major_axis = eigenvectors[:, np.argmax(eigenvalues)]
    yaw = np.arctan2(major_axis[1], major_axis[0])
    
    cos_t, sin_t = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    aligned_xy = xy @ R.T
    dim_x = np.max(aligned_xy[:, 0]) - np.min(aligned_xy[:, 0])
    dim_y = np.max(aligned_xy[:, 1]) - np.min(aligned_xy[:, 1])
    
    if is_bev:
        return dim_x * dim_y
    else:
        dim_z = np.max(pts[:, 2]) - np.min(pts[:, 2])
        return dim_x * dim_y * dim_z

def optimal_volume_flat_clustering(parents, points_dict, vol_prior, is_bev=False):
    N = len(parents)
    if N == 0: return []
    
    children = [[] for _ in range(N)]
    roots = []
    for i, p in enumerate(parents):
        if p == -1: roots.append(i)
        else: children[p].append(i)
        
    dp_val = np.zeros(N)
    is_leaf_cut = np.zeros(N, dtype=bool)
    
    def dfs(v):
        child_sum = 0.0
        for c in children[v]:
            child_sum += dfs(c)
            
        pts = points_dict.get(v, np.empty((0, 3)))
        if len(pts) >= 3:
            vol_p = compute_obb_volume(pts, is_bev)
            penalty = max(0, vol_p - vol_prior) * 10.0 + max(0, vol_prior - vol_p) * 0.1
            score_at_v = penalty
        else:
            score_at_v = 1e6
            
        if len(children[v]) == 0:
            dp_val[v] = score_at_v
            is_leaf_cut[v] = True
        else:
            if score_at_v <= child_sum:
                dp_val[v] = score_at_v
                is_leaf_cut[v] = True
            else:
                dp_val[v] = child_sum
                is_leaf_cut[v] = False
        return dp_val[v]

    for r in roots:
        dfs(r)
        
    V_opt = []
    def retrieve(v):
        if is_leaf_cut[v]:
            V_opt.append(v)
        else:
            for c in children[v]:
                retrieve(c)
                
    for r in roots:
        retrieve(r)
        
    node_hulls = {}
    for v in V_opt:
        pts = points_dict.get(v, np.empty((0, 3)))
        if len(pts) >= 3:
            hull = MultiPoint(pts[:, :2]).convex_hull
            if hull.geom_type in ['Point', 'LineString']: hull = hull.buffer(1e-5)
            node_hulls[v] = hull

    overlap_found = True
    max_iters = 10
    it = 0
    while overlap_found and it < max_iters:
        overlap_found = False
        it += 1
        
        valid_nodes = [v for v in V_opt if v in node_hulls]
        if len(valid_nodes) < 2: break
        
        geoms = [node_hulls[v] for v in valid_nodes]
        tree = STRtree(geoms)
        left, right = tree.query(geoms, predicate="intersects")
        
        for i, j in zip(left, right):
            if i >= j: continue
            u, v = valid_nodes[i], valid_nodes[j]
            
            can_split_u = len(children[u]) > 0
            can_split_v = len(children[v]) > 0
            
            if can_split_u and can_split_v:
                loss_u = sum(dp_val[c] for c in children[u]) - dp_val[u]
                loss_v = sum(dp_val[c] for c in children[v]) - dp_val[v]
                if loss_u < loss_v:
                    is_leaf_cut[u] = False
                else:
                    is_leaf_cut[v] = False
            elif can_split_u:
                is_leaf_cut[u] = False
            elif can_split_v:
                is_leaf_cut[v] = False
            else:
                continue
                
            overlap_found = True
            break
            
        if overlap_found:
            V_opt = []
            for r in roots:
                retrieve(r)
            for v in V_opt:
                if v not in node_hulls:
                    pts = points_dict.get(v, np.empty((0, 3)))
                    if len(pts) >= 3:
                        hull = MultiPoint(pts[:, :2]).convex_hull
                        if hull.geom_type in ['Point', 'LineString']: hull = hull.buffer(1e-5)
                        node_hulls[v] = hull

    return V_opt


def optimal_flat_clustering_cvx(parents, f_scores, points_dict, M):
    N = len(parents)
    if N == 0: return [], {}
    
    children = [[] for _ in range(N)]
    roots = []
    for i, p in enumerate(parents):
        if p == -1: roots.append(i)
        else: children[p].append(i)
        
    dp_val = np.zeros(N)
    best_m = np.zeros(N, dtype=int)
    is_leaf_cut = np.zeros(N, dtype=bool)
    
    # 1. Post-order traversal (bottom-up) iteration
    post_order = []
    stack = []
    for r in roots: stack.append(r)
    while stack:
        curr = stack.pop()
        post_order.append(curr)
        for c in children[curr]: stack.append(c)
            
    for v in reversed(post_order):
        child_sum = sum(dp_val[c] for c in children[v])
        
        best_score_at_v = -float('inf')
        b_m = 0
        for m in range(M + 1):
            if f_scores[v, m] > best_score_at_v:
                best_score_at_v = f_scores[v, m]
                b_m = m
                
        if len(children[v]) == 0:
            dp_val[v] = best_score_at_v
            best_m[v] = b_m
            is_leaf_cut[v] = True
        else:
            if best_score_at_v >= child_sum - 1e-6:
                dp_val[v] = best_score_at_v
                best_m[v] = b_m
                is_leaf_cut[v] = True
            else:
                dp_val[v] = child_sum
                is_leaf_cut[v] = False
                
    V_opt = []
    labels_opt = {}
    
    # 2. Top-down retrieval iteration
    stack = []
    for r in roots: stack.append(r)
    while stack:
        v = stack.pop()
        if is_leaf_cut[v]:
            V_opt.append(v)
            labels_opt[v] = best_m[v]
        else:
            for c in children[v]: stack.append(c)

    node_hulls = {}
    for v in V_opt:
        pts = points_dict.get(v, np.empty((0, 3)))
        if len(pts) >= 3:
            hull = MultiPoint(pts[:, :2]).convex_hull
            if hull.geom_type in ['Point', 'LineString']: hull = hull.buffer(1e-5)
            node_hulls[v] = hull

    # FAST CONFLICT RESOLUTION (O(1) without tree shatter)
    valid_nodes = [v for v in V_opt if v in node_hulls]
    if len(valid_nodes) >= 2:
        geoms = [node_hulls[v] for v in valid_nodes]
        tree = STRtree(geoms)
        left, right = tree.query(geoms, predicate="intersects")
        
        conflicts = set()
        for i, j in zip(left, right):
            if i >= j: continue
            u, v = valid_nodes[i], valid_nodes[j]
            
            if labels_opt[u] == labels_opt[v]:
                continue # Same track, no conflict
            
            conflicts.add(u)
            conflicts.add(v)
            
        for v in conflicts:
            labels_opt[v] = 0 # Trim conflicting nodes (they go to Passe 2)

    return V_opt, labels_opt

def compute_det(pts, device):
    c = np.mean(pts, axis=0)
    xy = pts[:, :2]
    if len(xy) > 2:
        cov = np.cov(xy.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        major_axis = eigenvectors[:, np.argmax(eigenvalues)]
        yaw = np.arctan2(major_axis[1], major_axis[0])
    else:
        yaw = 0.0
    
    cos_t, sin_t = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    aligned_xy = xy @ R.T
    dim_x = np.max(aligned_xy[:, 0]) - np.min(aligned_xy[:, 0])
    dim_y = np.max(aligned_xy[:, 1]) - np.min(aligned_xy[:, 1])
    dim_z = np.max(pts[:, 2]) - np.min(pts[:, 2])
    
    center_aligned_xy = np.array([np.min(aligned_xy[:, 0]) + dim_x/2, np.min(aligned_xy[:, 1]) + dim_y/2])
    cos_t2, sin_t2 = np.cos(yaw), np.sin(yaw)
    R_inv = np.array([[cos_t2, -sin_t2], [sin_t2, cos_t2]])
    center_xy = center_aligned_xy @ R_inv.T
    
    # We use the bounding box center for stability!
    c[0], c[1] = center_xy[0], center_xy[1]
    c[2] = np.min(pts[:, 2]) + dim_z/2
    
    dim = np.array([dim_x, dim_y, dim_z])
    dim = np.maximum(dim, [0.1, 0.1, 0.1])
    
    return {
        "centroid": c,
        "dimensions": dim,
        "yaw": yaw,
        "points_gpu": torch.tensor(pts, device=device, dtype=torch.float32)
    }

class Track:
    def __init__(self, track_id: int, semantic_class: int, det: dict, device: str = 'cuda', initial_velocity=None, initial_yaw_rate=0.0):
        self.track_id = track_id
        self.semantic_class = semantic_class
        self.device = device

        self.age_occlusion = 0
        self.age_total = 0
        self.hits = 1
        self.state = "Unconfirmed"
        c, dim, yaw = det["centroid"], det["dimensions"], det["yaw"]

        self.x = np.array([c[0], c[1], c[2], 0.0, 0.0, 0.0], dtype=float)
        if initial_velocity is not None:
            self.x[3:6] = initial_velocity
        self.P = np.eye(6) * 1.0
        self.P[3:6, 3:6] *= 10.0

        self.H = np.zeros((3, 6))
        self.H[0:3, 0:3] = np.eye(3)
        self.R = np.eye(3) * 0.5
        self.Q = np.eye(6) * 0.05
        self.Q[3:6, 3:6] = np.eye(3) * 0.05

        self.L, self.W, self.H_dim = dim[0], dim[1], dim[2]
        self.yaw = yaw
        self.yaw_rate = initial_yaw_rate

        self.last_points_gpu = det["points_gpu"].clone()
        self.pred_points_gpu = self.last_points_gpu.clone()

    @property
    def velocity(self):
        return self.x[3:6]

    def predict(self, dt: float):
        self.age_occlusion += 1
        self.age_total += 1

        F = np.eye(6)
        F[0, 3], F[1, 4], F[2, 5] = dt, dt, dt
        self.x = F @ self.x
        
        # Garde-fou : Limiter la vitesse à 35 m/s max pour éviter les projections aberrantes
        speed = np.linalg.norm(self.x[3:6])
        if speed > 35.0:
            self.x[3:6] = self.x[3:6] * (35.0 / speed)
            
        self.P = F @ self.P @ F.T + self.Q

        v_tensor = torch.tensor(self.x[3:6], device=self.device, dtype=torch.float32)
        total_dt = dt * self.age_occlusion

        # Réintégration sécurisée de la rotation (theta)
        # Clipping physique à +/- 0.1 rad (6° par frame) pour éviter l'instabilité
        theta = np.clip(self.yaw_rate * dt, -0.1, 0.1)
        if np.abs(theta) > 0.005: # Seuil minimal pour éviter le jitter (0.3°)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            R = torch.tensor([[cos_t, -sin_t, 0],
                              [sin_t,  cos_t, 0],
                              [0,      0,     1]], device=self.device, dtype=torch.float32)
            c_tensor = torch.mean(self.last_points_gpu, dim=0)
            centered_points = self.last_points_gpu - c_tensor
            rotated_points = torch.matmul(centered_points, R.T) + c_tensor
        else:
            rotated_points = self.last_points_gpu

        self.pred_points_gpu = rotated_points + v_tensor * total_dt

    def update(self, det: dict, dt: float = 0.1):
        self.hits += 1
        if self.hits >= 3:
            self.state = "Confirmed"
        elapsed_t = dt * max(1, self.age_occlusion)
        self.age_occlusion = 0
        c, dim, yaw = det["centroid"], det["dimensions"], det["yaw"]

        dyaw = np.arctan2(np.sin(yaw - self.yaw), np.cos(yaw - self.yaw))
        if dyaw > np.pi / 2:
            dyaw -= np.pi
        elif dyaw < -np.pi / 2:
            dyaw += np.pi
        if np.abs(dyaw) > np.pi / 2:
            dyaw = dyaw - np.sign(dyaw) * np.pi

        measured_yaw_rate = dyaw / elapsed_t
        alpha_yaw_rate = 0.25
        self.yaw_rate = (1 - alpha_yaw_rate) * self.yaw_rate + alpha_yaw_rate * measured_yaw_rate

        z = np.array([c[0], c[1], c[2]])
        S = self.H @ self.P @ self.H.T + self.R
        K_gain = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K_gain @ (z - self.H @ self.x)
        self.P = (np.eye(6) - K_gain @ self.H) @ self.P

        alpha = 0.2
        self.L = (1 - alpha) * self.L + alpha * dim[0]
        self.W = (1 - alpha) * self.W + alpha * dim[1]
        self.H_dim = (1 - alpha) * self.H_dim + alpha * dim[2]

        self.yaw = yaw
        self.last_points_gpu = det["points_gpu"].clone()
        self.pred_points_gpu = self.last_points_gpu.clone()

def solve_uot_sinkhorn_gpu(C: torch.Tensor, a: torch.Tensor, b: torch.Tensor, epsilon: float, tau1: float, tau2: float, iters: int = 15) -> torch.Tensor:
    K = torch.exp(-C / epsilon) 
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    exp_u = tau1 / (tau1 + epsilon)
    exp_v = tau2 / (tau2 + epsilon)
    for _ in range(iters):
        u = (a / (torch.matmul(K, v) + 1e-12)) ** exp_u
        v = (b / (torch.matmul(K.T, u) + 1e-12)) ** exp_v
    P = u.unsqueeze(1) * K * v.unsqueeze(0)
    return P

def compute_uot_matrix(tracks, cloud_pts, prior):
    if not tracks or len(cloud_pts) == 0:
        return np.zeros((len(cloud_pts), 0))
        
    M = len(tracks)
    N = len(cloud_pts)
    cloud_pts_gpu = torch.tensor(cloud_pts, device=DEVICE, dtype=torch.float32)
    
    pred_clouds = [tr.pred_points_gpu for tr in tracks]
    lengths = [cloud.shape[0] for cloud in pred_clouds]
    mega_pred_cloud = torch.cat(pred_clouds, dim=0)
    K_total = mega_pred_cloud.shape[0]
    
    C_matrix = torch.cdist(cloud_pts_gpu, mega_pred_cloud, p=2)**2
    gate_dist = prior.get("match_threshold", prior.get("max_speed", 20.0) * 0.1 * 2.0)
    C_matrix[C_matrix > gate_dist**2] = float('inf')
    
    a_f = torch.ones(N, device=DEVICE)
    b_f = torch.ones(K_total, device=DEVICE)
    
    tau_min = 1.0
    tau = tau_min + prior.get("max_speed", 20.0) * 0.1
    P_micro = solve_uot_sinkhorn_gpu(C_matrix, a_f, b_f, epsilon=0.2, tau1=tau, tau2=tau)
    P_micro_cpu = P_micro.cpu().numpy()
    
    V = np.zeros((N, M), dtype=np.float32)
    start_idx = 0
    for m, length in enumerate(lengths):
        V[:, m] = np.sum(P_micro_cpu[:, start_idx:start_idx+length], axis=1)
        start_idx += length
        
    return V

def generate_f_scores(tree_nodes, all_points, node_start, node_end, x_matrix, tau, M):
    N_nodes = len(tree_nodes)
    f_scores = np.zeros((N_nodes, M + 1))
    points_dict = {}
    for v in range(N_nodes):
        start = node_start[v]
        end = node_end[v]
        pts_indices = all_points[start:end]
        points_dict[v] = CURRENT_POINTS_3D[pts_indices]
        if len(pts_indices) == 0: continue
        x_v = np.sum(x_matrix[pts_indices], axis=0)
        n = len(pts_indices)
        sum_x = np.sum(x_v)
        f_scores[v, 0] = -tau * sum_x
        for m in range(1, M + 1):
            f_scores[v, m] = tau * (x_v[m-1] - n) - tau * (sum_x - x_v[m-1])
    return f_scores, points_dict

ALPINE_PRIORS = {
    1: {"name": "car",           "L": 4.5,  "W": 1.85, "H": 1.6,  "max_speed": 35.0, "match_threshold": 8.0, "tau": 4.0, "min_hits": 1},
    2: {"name": "bicycle",       "L": 1.8,  "W": 0.6,  "H": 1.1,  "max_speed": 15.0, "match_threshold": 3.0, "tau": 1.5, "min_hits": 3},
    3: {"name": "motorcycle",    "L": 2.2,  "W": 0.9,  "H": 1.3,  "max_speed": 35.0, "match_threshold": 3.0, "tau": 1.5, "min_hits": 3},
    4: {"name": "truck",         "L": 10.0, "W": 2.6,  "H": 3.5,  "max_speed": 30.0, "match_threshold": 30.0, "tau": 15.0, "min_hits": 1},
    5: {"name": "other-vehicle", "L": 12.0, "W": 2.6,  "H": 3.5,  "max_speed": 30.0, "match_threshold": 30.0, "tau": 15.0, "min_hits": 1},
    6: {"name": "person",        "L": 0.6,  "W": 0.6,  "H": 1.75, "max_speed": 5.0,  "match_threshold": 2.0, "tau": 1.0, "min_hits": 3},
    7: {"name": "bicyclist",     "L": 1.8,  "W": 0.75, "H": 1.8,  "max_speed": 15.0, "match_threshold": 3.0, "tau": 1.5, "min_hits": 3},
    8: {"name": "motorcyclist",  "L": 2.2,  "W": 0.9,  "H": 1.8,  "max_speed": 35.0, "match_threshold": 3.0, "tau": 1.5, "min_hits": 3},
}

# Initialisation
labels_pred = np.full(len(X_clustering), -1, dtype=int)
raw_labels = np.full(len(X_clustering), -1, dtype=int)
next_raw_id = 1
next_track_id = 1
tracks_dict = {} # {track_id: SimpleKalmanTrack}


def perform_volume_clustering(v_root, parents, points_dict, res_tree, vol_prior, is_bev, global_cl_indices, CURRENT_POINTS_3D, sem_cl):
    global next_raw_id, next_track_id, labels_pred, raw_labels, tracks_dict
    
    descendants = [v_root]
    stack = [v_root]
    while stack:
        curr = stack.pop()
        for i, p in enumerate(parents):
            if p == curr:
                stack.append(i)
                descendants.append(i)
                
    old_to_new = {old: new for new, old in enumerate(descendants)}
    new_parents = [-1] * len(descendants)
    new_points_dict = {}
    for old in descendants:
        new = old_to_new[old]
        p = parents[old]
        if p in old_to_new and old != v_root:
            new_parents[new] = old_to_new[p]
        new_points_dict[new] = points_dict[old]
        
    V_opt_vol = optimal_volume_flat_clustering(new_parents, new_points_dict, vol_prior, is_bev)
    
    assigned_local = []
    for new_v in V_opt_vol:
        old_v = descendants[new_v]
        start = res_tree['node_to_points_start'][old_v]
        end = res_tree['node_to_points_end'][old_v]
        pts_idx_local = res_tree['all_points'][start:end]
        if len(pts_idx_local) >= MIN_CLUSTER_SIZE:
            det = compute_det(CURRENT_POINTS_3D[pts_idx_local], DEVICE)
            new_track = Track(next_track_id, sem_cl, det, device=DEVICE)
            tracks_dict[next_track_id] = new_track
            labels_pred[global_cl_indices[pts_idx_local]] = next_track_id
            if TRACKING_VERBOSE: print(f"        -> [Vol Cluster] Piste Unconfirmed {next_track_id} créée ({len(pts_idx_local)} points).")
            raw_labels[global_cl_indices[pts_idx_local]] = next_raw_id
            next_raw_id += 1
            next_track_id += 1
            assigned_local.extend(pts_idx_local.tolist())
    return assigned_local

frames = np.unique(Time_idx)


for t in frames:
    f_mask = (Time_idx == t)
    global_indices = np.where(f_mask)[0]
    if TRACKING_VERBOSE: print(f"\n{'='*60}\nFRAME {t} (Indices globaux: {len(global_indices)})\n{'='*60}")
    
    for tr in tracks_dict.values():
        tr.predict(0.1)
        
    classes_in_frame = np.unique(Y_sem[f_mask])
    
    for sem_cl in classes_in_frame:
        if sem_cl not in THINGS_CLASSES and SEMANTIC_MODE == "Oracle": continue
        
        cl_mask = (Y_sem[f_mask] == sem_cl)
        global_cl_indices = global_indices[cl_mask]
        X_cl = X_clustering[f_mask][cl_mask]
        if len(X_cl) < MIN_CLUSTER_SIZE: continue
        
        prior = ALPINE_PRIORS.get(sem_cl, {"L": 1.0, "W": 1.0, "H": 1.0, "tau": 2.0, "name": f"cl_{sem_cl}"})
        tau = prior["tau"]
        CURRENT_POINTS_3D = X_4d[global_cl_indices][:, :3]
        
        if TRACKING_VERBOSE: print(f"  [Process] Classe {prior.get('name')} ({sem_cl}) : {len(X_cl)} points.")
        # --- PASSE 1: CONFIRMED TRACKS ---
        confirmed_tracks = [tr for tr in tracks_dict.values() if tr.state == "Confirmed" and tr.age_occlusion <= 5 and tr.semantic_class == sem_cl]
        M_conf = len(confirmed_tracks)
        if TRACKING_VERBOSE: print(f"    - Passe 1 (Confirmed) : {M_conf} pistes actives.")
        
        assigned_points_conf = set()
        spatial_X = X_cl[:, :2] if APPLY_BEV else X_cl[:, :3]
        
        if M_conf > 0:
            global_track_assignments_conf = defaultdict(list)
            x_matrix_conf = compute_uot_matrix(confirmed_tracks, CURRENT_POINTS_3D, prior)
            
            clusterer = HGPClusterer(K=K, min_cluster_size=MIN_CLUSTER_SIZE, method=prior["L"]*DBSCAN_FACTOR, expZ=EXP_Z, backend=BACKEND)
            try:
                clusterer.fit(spatial_X)
            except Exception as e: print(f"Error: {e}"); continue
            
            if hasattr(clusterer, 'forest_') and clusterer.forest_:
                for comp in clusterer.forest_:
                    Z = comp['Z']
                    faces_cc = comp['faces_cc']
                    
                    try:
                        from hgp_clusterer.clustering import GetClusters
                        res_tree = GetClusters(Z, method=prior["L"]*DBSCAN_FACTOR, Face_to_points=faces_cc, whole_tree=True)
                    except Exception as e: print(f"Error: {e}"); continue
                        
                    if 'tree_nodes' not in res_tree: continue
                        
                    parents = res_tree['tree_parents']
                    f_scores, points_dict = generate_f_scores(res_tree['tree_nodes'], res_tree['all_points'], res_tree['node_to_points_start'], res_tree['node_to_points_end'], x_matrix_conf, tau, M_conf)
                    
                    try:
                        V_opt, labels_opt = optimal_flat_clustering_cvx(parents, f_scores, points_dict, M_conf)
                        if TRACKING_VERBOSE: print(f"      * [MILP Confirmed] Solved pour composante de {len(parents)} noeuds. {len(V_opt)} noeuds gardés.")
                    except Exception as e: 
                        import traceback
                        if TRACKING_VERBOSE: print(f"      * [MILP Error] {e}\n{traceback.format_exc()}")
                        continue
                        
                    roots = [i for i, p in enumerate(parents) if p == -1]
                    for r in roots:
                        stack = [r]
                        descendants = set([r])
                        while stack:
                            curr = stack.pop()
                            for i, p in enumerate(parents):
                                if p == curr:
                                    stack.append(i)
                                    descendants.add(i)
                                    
                        v_opt_in_tree = [v for v in descendants if v in V_opt]
                        subtree_labels = set(labels_opt.get(v, 0) for v in v_opt_in_tree)
                        
                        if len(subtree_labels) == 1 and 0 in subtree_labels:
                            pass # Pur bruit, on laisse pour Unconfirmed
                        else:
                            # L'arbre entier a reçu au moins un label > 0.
                            # On l'ignore *intégralement* pour la Passe 2.
                            start_root = res_tree['node_to_points_start'][r]
                            end_root = res_tree['node_to_points_end'][r]
                            pts_idx_tree = res_tree['all_points'][start_root:end_root]
                            assigned_points_conf.update(pts_idx_tree.tolist())
                            
                            tree_assignments_m = defaultdict(list)
                            for v in v_opt_in_tree:
                                m = labels_opt[v]
                                start = res_tree['node_to_points_start'][v]
                                end = res_tree['node_to_points_end'][v]
                                pts_idx = res_tree['all_points'][start:end]
                                if len(pts_idx) == 0: continue
                                
                                if m > 0:
                                    tree_assignments_m[m].append(pts_idx)
                                elif m == 0:
                                    pass # Deja ajoute
                                    
                            for m, list_of_pts in tree_assignments_m.items():
                                all_pts_m = np.concatenate(list_of_pts)
                                x_v = np.sum(x_matrix_conf[all_pts_m], axis=0)
                                uot_weight = x_v[m-1]
                                global_track_assignments_conf[m].append({'pts_idx': all_pts_m, 'uot_weight': uot_weight})

            track_updates_conf = defaultdict(list)
            for m, candidates in global_track_assignments_conf.items():
                if not candidates: continue
                candidates.sort(key=lambda x: x['uot_weight'], reverse=True)
                track = confirmed_tracks[m-1]
                main_pts = candidates[0]['pts_idx']
                track_updates_conf[track.track_id].append(main_pts)
                labels_pred[global_cl_indices[main_pts]] = track.track_id
                assigned_points_conf.update(main_pts.tolist())
                raw_labels[global_cl_indices[main_pts]] = next_raw_id
                next_raw_id += 1
                
                for cand in candidates[1:]:
                    pts_idx_global = cand['pts_idx']
                    if len(pts_idx_global) < MIN_CLUSTER_SIZE: continue
                    det = compute_det(CURRENT_POINTS_3D[pts_idx_global], DEVICE)
                    new_track = Track(next_track_id, sem_cl, det, device=DEVICE, initial_velocity=track.velocity, initial_yaw_rate=track.yaw_rate)
                    tracks_dict[next_track_id] = new_track
                    labels_pred[global_cl_indices[pts_idx_global]] = next_track_id
                    if TRACKING_VERBOSE: print(f"        -> [Split Conf] Piste {track.track_id} scindée en Unconf {next_track_id} ({len(pts_idx_global)} pts).")
                    raw_labels[global_cl_indices[pts_idx_global]] = next_raw_id
                    next_raw_id += 1
                    next_track_id += 1
                    assigned_points_conf.update(pts_idx_global.tolist())

            for track_id, pts_list in track_updates_conf.items():
                all_pts_idx = np.concatenate(pts_list)
                track = next((t for t in confirmed_tracks if t.track_id == track_id), None)
                if track:
                    det = compute_det(CURRENT_POINTS_3D[all_pts_idx], DEVICE)
                    track.update(det, dt=0.1)
                    if TRACKING_VERBOSE: print(f"        -> [Update] Piste {track.track_id} reçoit {len(all_pts_idx)} points.")
    
        # --- PASSE 2: UNCONFIRMED TRACKS ET NAISSANCES ---
        unassigned_mask = np.ones(len(CURRENT_POINTS_3D), dtype=bool)
        unassigned_mask[list(assigned_points_conf)] = False
        
        points_idx_unconf = np.where(unassigned_mask)[0]
        if TRACKING_VERBOSE: print(f"    - Passe 2 (Unconfirmed) : {len(points_idx_unconf)} points restants.")
        if len(points_idx_unconf) < MIN_CLUSTER_SIZE:
            continue
            
        spatial_X_unconf = spatial_X[points_idx_unconf]
        CURRENT_POINTS_3D_UNCONF = CURRENT_POINTS_3D[points_idx_unconf]
        
        unconfirmed_tracks = [tr for tr in tracks_dict.values() if tr.state == "Unconfirmed" and tr.semantic_class == sem_cl]
        M_unconf = len(unconfirmed_tracks)
        
        if M_unconf == 0:
            clusterer_unconf = HGPClusterer(K=K, min_cluster_size=MIN_CLUSTER_SIZE, method=prior["L"]*DBSCAN_FACTOR, expZ=EXP_Z, backend=BACKEND)
            try:
                clusterer_unconf.fit(spatial_X_unconf)
            except Exception as e: print(f"Error: {e}"); continue
            
            if hasattr(clusterer_unconf, 'forest_') and clusterer_unconf.forest_:
                for comp in clusterer_unconf.forest_:
                    Z = comp['Z']
                    faces_cc = comp['faces_cc']
                    try:
                        from hgp_clusterer.clustering import GetClusters
                        res_tree = GetClusters(Z, method=prior["L"]*DBSCAN_FACTOR, Face_to_points=faces_cc, whole_tree=True)
                    except Exception as e: print(f"Error: {e}"); continue
                        
                    if 'tree_nodes' not in res_tree: continue
                    parents = res_tree['tree_parents']
                    N_nodes = len(res_tree['tree_nodes'])
                    points_dict = {}
                    for v in range(N_nodes):
                        start = res_tree['node_to_points_start'][v]
                        end = res_tree['node_to_points_end'][v]
                        pts_indices = res_tree['all_points'][start:end]
                        points_dict[v] = CURRENT_POINTS_3D_UNCONF[pts_indices]
                        
                    roots = [i for i, p in enumerate(parents) if p == -1]
                    for r in roots:
                        vol_prior = prior["L"] * prior["W"] * (1.0 if APPLY_BEV else prior.get("H", 1.5)) * 0.8
                        perform_volume_clustering(r, parents, points_dict, res_tree, vol_prior, APPLY_BEV, global_cl_indices[points_idx_unconf], CURRENT_POINTS_3D_UNCONF, sem_cl)
        else:
            global_track_assignments_unconf = defaultdict(list)
            x_matrix_unconf = compute_uot_matrix(unconfirmed_tracks, CURRENT_POINTS_3D_UNCONF, prior)
            
            clusterer_unconf = HGPClusterer(K=K, min_cluster_size=MIN_CLUSTER_SIZE, method=prior["L"]*DBSCAN_FACTOR, expZ=EXP_Z, backend=BACKEND)
            try:
                clusterer_unconf.fit(spatial_X_unconf)
            except Exception as e: print(f"Error: {e}"); continue
            
            if hasattr(clusterer_unconf, 'forest_') and clusterer_unconf.forest_:
                for comp in clusterer_unconf.forest_:
                    Z = comp['Z']
                    faces_cc = comp['faces_cc']
                    
                    try:
                        from hgp_clusterer.clustering import GetClusters
                        res_tree = GetClusters(Z, method=prior["L"]*DBSCAN_FACTOR, Face_to_points=faces_cc, whole_tree=True)
                    except Exception as e: print(f"Error: {e}"); continue
                        
                    if 'tree_nodes' not in res_tree: continue
                        
                    parents = res_tree['tree_parents']
                    CURRENT_POINTS_3D_TMP = CURRENT_POINTS_3D
                    CURRENT_POINTS_3D = CURRENT_POINTS_3D_UNCONF
                    f_scores, points_dict = generate_f_scores(res_tree['tree_nodes'], res_tree['all_points'], res_tree['node_to_points_start'], res_tree['node_to_points_end'], x_matrix_unconf, tau, M_unconf)
                    CURRENT_POINTS_3D = CURRENT_POINTS_3D_TMP
                    
                    try:
                        V_opt, labels_opt = optimal_flat_clustering_cvx(parents, f_scores, points_dict, M_unconf)
                        if TRACKING_VERBOSE: print(f"      * [MILP Unconf] Solved pour composante de {len(parents)} noeuds. {len(V_opt)} noeuds gardés.")
                    except Exception as e: 
                        import traceback
                        if TRACKING_VERBOSE: print(f"      * [MILP Error] {e}\n{traceback.format_exc()}")
                        continue
                        
                    roots = [i for i, p in enumerate(parents) if p == -1]
                    for r in roots:
                        stack = [r]
                        descendants = set([r])
                        while stack:
                            curr = stack.pop()
                            for i, p in enumerate(parents):
                                if p == curr:
                                    stack.append(i)
                                    descendants.add(i)
                                    
                        v_opt_in_tree = [v for v in descendants if v in V_opt]
                        subtree_labels = set(labels_opt.get(v, 0) for v in v_opt_in_tree)
                        
                        if len(subtree_labels) == 1 and 0 in subtree_labels:
                            vol_prior = prior["L"] * prior["W"] * (1.0 if APPLY_BEV else prior.get("H", 1.5)) * 0.8
                            perform_volume_clustering(r, parents, points_dict, res_tree, vol_prior, APPLY_BEV, global_cl_indices[points_idx_unconf], CURRENT_POINTS_3D_UNCONF, sem_cl)
                        else:
                            tree_assignments_m = defaultdict(list)
                            for v in v_opt_in_tree:
                                m = labels_opt[v]
                                start = res_tree['node_to_points_start'][v]
                                end = res_tree['node_to_points_end'][v]
                                pts_idx_local = res_tree['all_points'][start:end]
                                if len(pts_idx_local) == 0: continue
                                
                                if m > 0:
                                    tree_assignments_m[m].append(pts_idx_local)
                                elif m == 0:
                                    pass # Ignored due to conflict

                            for m, list_of_pts in tree_assignments_m.items():
                                all_pts_m_local = np.concatenate(list_of_pts)
                                x_v = np.sum(x_matrix_unconf[all_pts_m_local], axis=0)
                                uot_weight = x_v[m-1]
                                global_track_assignments_unconf[m].append({'pts_idx': all_pts_m_local, 'uot_weight': uot_weight})

            track_updates_unconf = defaultdict(list)
            for m, candidates in global_track_assignments_unconf.items():
                if not candidates: continue
                candidates.sort(key=lambda x: x['uot_weight'], reverse=True)
                track = unconfirmed_tracks[m-1]
                main_pts_local = candidates[0]['pts_idx']
                track_updates_unconf[track.track_id].append(main_pts_local)
                pts_idx_global = points_idx_unconf[main_pts_local]
                labels_pred[global_cl_indices[pts_idx_global]] = track.track_id
                raw_labels[global_cl_indices[pts_idx_global]] = next_raw_id
                next_raw_id += 1
                
                for cand in candidates[1:]:
                    split_pts_local = cand['pts_idx']
                    pts_idx_global = points_idx_unconf[split_pts_local]
                    if len(pts_idx_global) < MIN_CLUSTER_SIZE: continue
                    det = compute_det(CURRENT_POINTS_3D_UNCONF[split_pts_local], DEVICE)
                    new_track = Track(next_track_id, sem_cl, det, device=DEVICE, initial_velocity=track.velocity, initial_yaw_rate=track.yaw_rate)
                    tracks_dict[next_track_id] = new_track
                    labels_pred[global_cl_indices[pts_idx_global]] = next_track_id
                    if TRACKING_VERBOSE: print(f"        -> [Split Unconf] Piste {track.track_id} scindée en Unconf {next_track_id} ({len(pts_idx_global)} pts).")
                    raw_labels[global_cl_indices[pts_idx_global]] = next_raw_id
                    next_raw_id += 1
                    next_track_id += 1
                    
            for track_id, pts_list in track_updates_unconf.items():
                all_pts_idx_local = np.concatenate(pts_list)
                pts_idx_global = points_idx_unconf[all_pts_idx_local]
                track = next((t for t in unconfirmed_tracks if t.track_id == track_id), None)
                if track:
                    det = compute_det(CURRENT_POINTS_3D_UNCONF[all_pts_idx_local], DEVICE)
                    track.update(det, dt=0.1)
                    if track.state == "Confirmed":
                        if TRACKING_VERBOSE: print(f"        -> [Promoted] Piste {track.track_id} confirmée avec {len(pts_idx_global)} points.")
                    else:
                        if TRACKING_VERBOSE: print(f"        -> [Update Unconf] Piste {track.track_id} survit avec {len(pts_idx_global)} points.")

        # --- PASSE 3: REPECHAGE (Ré-association des pistes occluses) ---
        occluded_tracks = [tr for tr in tracks_dict.values() if tr.state == "Confirmed" and tr.age_occlusion > 0 and tr.semantic_class == sem_cl]
        candidate_tracks = [tr for tr in tracks_dict.values() if tr.state == "Unconfirmed" and tr.age_occlusion == 0 and tr.semantic_class == sem_cl]
        
        if occluded_tracks and candidate_tracks:
            from scipy.optimize import linear_sum_assignment
            cost_matrix = np.zeros((len(occluded_tracks), len(candidate_tracks)))
            for i, occ_tr in enumerate(occluded_tracks):
                for j, cand_tr in enumerate(candidate_tracks):
                    dist = np.linalg.norm(occ_tr.pred_points_gpu.mean(dim=0).cpu().numpy()[:2] - cand_tr.x[:2])
                    cost_matrix[i, j] = dist
                    
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for i, j in zip(row_ind, col_ind):
                occ_tr = occluded_tracks[i]
                cand_tr = candidate_tracks[j]
                
                if cost_matrix[i, j] < prior.get("max_speed", 20.0) * 0.1 * occ_tr.age_occlusion * 1.5: # 50% tolérance supp pour l'occlusion
                    # Transfert d'identité (Merge)
                    occ_tr.x = cand_tr.x
                    occ_tr.P = cand_tr.P
                    occ_tr.yaw = cand_tr.yaw
                    occ_tr.yaw_rate = cand_tr.yaw_rate
                    occ_tr.L = cand_tr.L
                    occ_tr.W = cand_tr.W
                    occ_tr.H_dim = cand_tr.H_dim
                    occ_tr.last_points_gpu = cand_tr.last_points_gpu
                    occ_tr.pred_points_gpu = cand_tr.pred_points_gpu
                    occ_tr.age_occlusion = 0
                    occ_tr.hits += cand_tr.hits
                    
                    # Remplacement des IDs
                    mask = (labels_pred == cand_tr.track_id)
                    labels_pred[mask] = occ_tr.track_id
                    
                    if TRACKING_VERBOSE: print(f"        -> [Repechage] Piste Unconf {cand_tr.track_id} fusionnée dans Piste Occluse {occ_tr.track_id} (Dist: {cost_matrix[i,j]:.2f}m)")
                    del tracks_dict[cand_tr.track_id]
                                    
    to_delete = []
    for tid, tr in tracks_dict.items():
        if tr.age_occlusion > 5:
            to_delete.append(tid)
        elif tr.state == "Unconfirmed" and tr.age_occlusion > 0:
            to_delete.append(tid)
            
    for tid in to_delete:
        del tracks_dict[tid]

print(f"\nTracking terminé. Total IDs uniques : {next_track_id - 1}")





try:
    from google.colab import output
    output.enable_custom_widget_manager()
except ImportError:
    pass
# @title 5.1 Évaluation Officielle SemanticKITTI (PQ, SQ, RQ)
import os
import shutil
import numpy as np
import yaml

if X_clustering is not None and len(X_clustering) > 0:
    print("Préparation des fichiers pour l'évaluation officielle (semantic-kitti-api)...")
    
    eval_dir = "/content/eval_data"
    pred_dir = "/content/eval_predictions"
    seq_str = f"{SEQUENCE_TO_TEST:02d}"
    
    gt_labels_dir = os.path.join(eval_dir, "sequences", seq_str, "labels")
    pred_labels_dir = os.path.join(pred_dir, "sequences", seq_str, "predictions")
    
    # Nettoyage précédent éventuel
    shutil.rmtree(eval_dir, ignore_errors=True)
    shutil.rmtree(pred_dir, ignore_errors=True)
    
    os.makedirs(gt_labels_dir, exist_ok=True)
    os.makedirs(pred_labels_dir, exist_ok=True)
    
    # Création d'une configuration personnalisée pour n'évaluer que cette séquence
    custom_cfg_path = "/content/custom_eval_config.yaml"
    with open("/content/semantic-kitti-api/config/semantic-kitti.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['split']['valid'] = [SEQUENCE_TO_TEST]
    with open(custom_cfg_path, 'w') as f:
        yaml.dump(cfg, f)
    
    # On reconstruit les labels prédits frame par frame
    actual_num_frames = len(loader) - START_FRAME if NUM_FRAMES == -1 else NUM_FRAMES
    for i in range(actual_num_frames):
        idx = START_FRAME + i
        if idx >= len(loader): break
        
        # Copie du GT
        gt_file = loader.label_files[idx]
        shutil.copy(gt_file, os.path.join(gt_labels_dir, os.path.basename(gt_file)))
        
        # Récupération des labels originaux pour garder la sémantique de fond
        s_raw, _ = loader.get_labels(idx)
        s_mapped = LABEL_MAP_ARRAY[s_raw]
        
        if SEMANTIC_MODE == "Oracle":
            mask = np.array([sem in THINGS_CLASSES for sem in s_mapped])
            frame_indices = np.where(mask)[0]
        else:
            frame_indices = np.arange(len(s_raw))
            
        # Initialisation de la prédiction
        pred_label = s_raw.astype(np.uint32)
        
        mask_time = (Time_idx == i)
        inst_preds = labels_pred[mask_time]
        
        valid_inst_mask = inst_preds >= 0
        valid_local_indices = frame_indices[valid_inst_mask]
        # Offset +1 car l'ID 0 est réservé au background
        valid_inst_ids = inst_preds[valid_inst_mask] + 1 
        
        pred_label[valid_local_indices] = (s_raw[valid_local_indices] & 0xFFFF) | (valid_inst_ids.astype(np.uint32) << 16)
        
        # Sauvegarde
        pred_filename = os.path.join(pred_labels_dir, os.path.basename(gt_file))
        pred_label.tofile(pred_filename)
        
    print("Fichiers de prédiction générés. Téléchargement des scripts d'évaluation 4D...")
    os.system("wget -q https://raw.githubusercontent.com/MehmetAygun/4D-PLS/master/utils/evaluate_4dpanoptic.py -O /content/semantic-kitti-api/evaluate_4dpanoptic.py")
    os.system("wget -q https://raw.githubusercontent.com/MehmetAygun/4D-PLS/master/utils/eval_np.py -O /content/semantic-kitti-api/auxiliary/eval_np_4d.py")
    os.system("sed -i 's/from eval_np import Panoptic4DEval/from auxiliary.eval_np_4d import Panoptic4DEval/' /content/semantic-kitti-api/evaluate_4dpanoptic.py")
    
    os.makedirs("/content/eval_output", exist_ok=True)
    # On lance le script officiel 4D sur ce mini-dataset de test et on capture la sortie
    import subprocess
    print("Évaluation en cours (cela peut prendre quelques minutes)...")
    eval_cmd = f"python3 /content/semantic-kitti-api/evaluate_4dpanoptic.py --dataset {eval_dir} --predictions {pred_dir} --split valid --data_cfg {custom_cfg_path}"
    res = subprocess.run(eval_cmd, shell=True, capture_output=True, text=True)
    
    print("\n--- RÉSULTATS DE L'ÉVALUATION LSTQ (4D) ---")
    print(res.stdout)
    if res.stderr:
        print("Erreurs :", res.stderr)
    
    print("\n--- RÉSULTATS NORMALISÉS 4D (Classes présentes uniquement) ---")
    try:
        import re
        # Extraction des tableaux Numpy de la sortie brute
        aq_match = re.search(r'Assoc:\s*\[(.*?)\]', res.stdout, re.DOTALL)
        iou_match = re.search(r'iou:\s*\[(.*?)\]', res.stdout, re.DOTALL)
        
        if aq_match and iou_match:
            aq_arr = [float(x) for x in aq_match.group(1).replace('\n', '').split()]
            iou_arr = [float(x) for x in iou_match.group(1).replace('\n', '').split()]
            
            present_sem_classes = np.unique(Y_sem)
            things_names_mapping = {1: "car", 2: "bicycle", 3: "motorcycle", 4: "truck", 5: "other-vehicle", 6: "person", 7: "bicyclist", 8: "motorcyclist"}
            present_things_ids = [c for c in present_sem_classes if c in things_names_mapping]
            
            if present_things_ids:
                aq_sum, iou_sum, lstq_sum = 0.0, 0.0, 0.0
                for cid in present_things_ids:
                    c_aq = aq_arr[cid]
                    c_iou = iou_arr[cid]
                    c_lstq = np.sqrt(c_aq * c_iou)
                    
                    aq_sum += c_aq
                    iou_sum += c_iou
                    lstq_sum += c_lstq
                    
                n = len(present_things_ids)
                print(f"Classes Things présentes ({n}) : {', '.join([things_names_mapping[c] for c in present_things_ids])}")
                print(f"LSTQ_things_normalized: {lstq_sum / n:.4f}")
                print(f"S_cls (IoU)_things_normalized: {iou_sum / n:.4f}")
                print(f"S_assoc (AQ)_things_normalized: {aq_sum / n:.4f}")
        else:
            print("Impossible de parser la sortie du script.")
    except Exception as e:
        print(f"Erreur lors du calcul normalisé : {e}")
else:
    print("Pas de données pour l'évaluation.")


# @title 6.1 Visualisation 4D (Monde LiDAR : X,Y = Sol | Z = Temps)
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

VIZ_START_FRAME = 0 # @param {type:"integer"}
VIZ_END_FRAME = 200 # @param {type:"integer"}

if X_4d is not None and len(X_4d) > 0:
    # Filtrage par frames pour l'affichage
    frame_mask = (Time_idx >= VIZ_START_FRAME) & (Time_idx <= VIZ_END_FRAME)
    X_4d_viz = X_4d[frame_mask]
    labels_pred_viz = labels_pred[frame_mask]
    Y_inst_viz = Y_inst[frame_mask]
    Y_sem_viz = Y_sem[frame_mask]
    raw_labels_viz = raw_labels[frame_mask]
    Time_idx_viz = Time_idx[frame_mask]
    
    # --- Calcul du Mapping IoU > 50% ---
    # --- Calcul du Mapping IoU > 50% sur l'ensemble des frames testées ---
    print("Calcul des correspondances (IoU > 50%) globales entre Prédictions et Vérité Terrain...")
    valid_mask_global = (Y_inst > -1) & (labels_pred > -1)
    gt_valid_global = Y_inst[valid_mask_global]
    pred_valid_global = labels_pred[valid_mask_global]
    
    df = pd.DataFrame({'gt': gt_valid_global, 'pred': pred_valid_global})
    pair_counts = df.groupby(['gt', 'pred']).size().reset_index(name='count')
    
    gt_counts = pd.Series(Y_inst[Y_inst > -1]).value_counts()
    pred_counts = pd.Series(labels_pred[labels_pred > -1]).value_counts()
    
    iou_map = {}
    for _, row in pair_counts.iterrows():
        gt_id, pred_id, count = int(row['gt']), int(row['pred']), int(row['count'])
        union = gt_counts.get(gt_id, 0) + pred_counts.get(pred_id, 0) - count
        iou = count / max(1, union)
        if iou > 0.5:
            iou_map[pred_id] = gt_id
            
    print(f"-> {len(iou_map)} pistes ont été matchées parfaitement avec la Ground Truth (IoU global > 50%)")
    # -----------------------------------
    
    max_points = 30000
    if len(X_4d_viz) > max_points:
        idx = np.random.choice(len(X_4d_viz), max_points, replace=False)
    else:
        idx = np.arange(len(X_4d_viz))
        
    X_v, L_v, GT_v, S_v, R_v, T_v = X_4d_viz[idx].copy(), labels_pred_viz[idx], Y_inst_viz[idx], Y_sem_viz[idx], raw_labels_viz[idx], Time_idx_viz[idx]
    
    THINGS_NAMES = {1: "car", 2: "bicycle", 3: "motorcycle", 4: "truck", 5: "other-vehicle", 6: "person", 7: "bicyclist", 8: "motorcyclist"}
    
    fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
                        subplot_titles=('Ground Truth', 'Raw Clustering', 'Panoptic Tracking'),
                        horizontal_spacing=0.02)
    
    def add_traces(ids, col, name_prefix):
        m = ids > -1
        if not np.any(m): return
        m_x, m_y, m_z, m_ids, m_s, m_gt, m_t = X_v[m, 0], X_v[m, 1], X_v[m, 3], ids[m], S_v[m], GT_v[m], T_v[m]
        
        if name_prefix == "Track":
            hover_texts = [f"Frame: {t}<br>Class: {THINGS_NAMES.get(s, 'unknown')}<br>Track ID: {i}<br>Matched GT: {iou_map.get(i, 'None')}" for t, s, i in zip(m_t, m_s, m_ids)]
            mapped_ids = np.array([iou_map.get(i, i + 500000) for i in m_ids])
            color_values = (mapped_ids * 97) % 256
        else:
            hover_texts = [f"Frame: {t}<br>Class: {THINGS_NAMES.get(s, 'unknown')}<br>ID: {i}" for t, s, i in zip(m_t, m_s, m_ids)]
            color_values = (m_ids * 97) % 256
        
        fig.add_trace(go.Scatter3d(
            x=m_x, y=m_y, z=m_z,
            mode='markers', 
            marker=dict(size=2, color=color_values, colorscale='Turbo', cmin=0, cmax=255), 
            name=name_prefix,
            text=hover_texts, hoverinfo='text+name'
        ), row=1, col=col)

    add_traces(GT_v, 1, "GT")
    add_traces(R_v, 2, "Raw")
    add_traces(L_v, 3, "Track")
    
    scene_config = dict(
        aspectmode='data',
        xaxis_title='X Monde (m)', yaxis_title='Y Monde (m)', zaxis_title='Temps (s)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    )
    
    fig.update_layout(title_text="Segmentation Panoptique 4D (Référentiel Monde LiDAR)", 
                      height=850, showlegend=False,
                      scene=scene_config, scene2=scene_config, scene3=scene_config)
    pass # fig.show()
else:
    print("Pas de données à afficher.")

# @title 6.1.bis Visualisation 3D d'une Frame Unique (Qualité Papier)
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FRAME_TO_SHOW = 0 # @param {type:"integer"}

# SemanticKITTI colors (BGR to RGB)
SEMANTIC_COLORS = {
  0 : [0, 0, 0],
  1 : [255, 0, 0],
  10: [100, 150, 245],
  11: [100, 230, 245],
  13: [100, 80, 250],
  15: [30, 60, 150],
  16: [0, 0, 255],
  18: [80, 30, 180],
  20: [0, 0, 255],
  30: [255, 30, 30],
  31: [255, 40, 200],
  32: [150, 30, 90],
  40: [255, 0, 255],
  44: [255, 150, 255],
  48: [75, 0, 75],
  49: [175, 0, 75],
  50: [255, 200, 0],
  51: [255, 120, 50],
  52: [255, 150, 0],
  60: [150, 255, 170],
  70: [0, 175, 0],
  71: [135, 60, 0],
  72: [150, 240, 80],
  80: [255, 240, 150],
  81: [255, 0, 255],
  99: [50, 255, 255],
  252: [100, 150, 245],
  256: [0, 0, 255],
  253: [255, 40, 200],
  254: [255, 30, 30],
  255: [150, 30, 90],
  257: [100, 80, 250],
  258: [80, 30, 180],
  259: [0, 0, 255]
}

SEMANTIC_NAMES = {
  0 : "unlabeled",
  1 : "outlier",
  10: "car",
  11: "bicycle",
  13: "bus",
  15: "motorcycle",
  16: "on-rails",
  18: "truck",
  20: "other-vehicle",
  30: "person",
  31: "bicyclist",
  32: "motorcyclist",
  40: "road",
  44: "parking",
  48: "sidewalk",
  49: "other-ground",
  50: "building",
  51: "fence",
  52: "other-structure",
  60: "lane-marking",
  70: "vegetation",
  71: "trunk",
  72: "terrain",
  80: "pole",
  81: "traffic-sign",
  99: "other-object",
  252: "moving-car",
  256: "moving-on-rails",
  253: "moving-bicyclist",
  254: "moving-person",
  255: "moving-motorcyclist",
  257: "moving-bus",
  258: "moving-truck",
  259: "moving-other-vehicle"
}

STUFF_CLASSES = [40, 44, 48, 49, 50, 51, 52, 60, 70, 71, 72, 80, 81, 99]

if X_4d is not None and len(X_4d) > 0:
    frame_mask = (Time_idx == FRAME_TO_SHOW)
    
    if np.any(frame_mask):
        X_frame = X_4d[frame_mask]
        labels_frame = labels_pred[frame_mask]
        Y_sem_frame = Y_sem[frame_mask]
        GT_frame = Y_inst[frame_mask]
        
        # Sous-échantillonnage optionnel pour affichage fluide
        max_points_frame = 100000
        if len(X_frame) > max_points_frame:
            idx = np.random.choice(len(X_frame), max_points_frame, replace=False)
            X_frame = X_frame[idx]
            labels_frame = labels_frame[idx]
            Y_sem_frame = Y_sem_frame[idx]
            GT_frame = GT_frame[idx]
            
        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                            subplot_titles=('Ground Truth', 'Clustering & Tracking'),
                            horizontal_spacing=0.02)
        
        def add_frame_traces(col, ids, is_gt=False):
            # STUFF points
            stuff_mask = np.isin(Y_sem_frame, STUFF_CLASSES)
            if np.any(stuff_mask):
                # Convert to rgba with 0.15 opacity for stuff
                stuff_colors = np.array([SEMANTIC_COLORS.get(c, [128, 128, 128]) for c in Y_sem_frame[stuff_mask]])
                stuff_colors_str = [f'rgba({r},{g},{b},0.15)' for r, g, b in stuff_colors]
                hover_texts = [f"Class: {SEMANTIC_NAMES.get(c, 'unknown')} ({c})" for c in Y_sem_frame[stuff_mask]]
                
                fig.add_trace(go.Scatter3d(
                    x=X_frame[stuff_mask, 0], y=X_frame[stuff_mask, 1], z=X_frame[stuff_mask, 2],
                    mode='markers',
                    marker=dict(size=1.5, color=stuff_colors_str),
                    name='STUFF',
                    text=hover_texts,
                    hoverinfo='text'
                ), row=1, col=col)
                
            # THINGS points
            things_mask = ~stuff_mask & (ids > -1)
            if np.any(things_mask):
                unique_ids = np.unique(ids[things_mask])
                for uid in unique_ids:
                    uid_mask = things_mask & (ids == uid)
                    if not np.any(uid_mask): continue
                    
                    # Récupération de la classe sémantique majoritaire
                    sem_class = np.bincount(Y_sem_frame[uid_mask]).argmax()
                    class_name = SEMANTIC_NAMES.get(sem_class, 'unknown')
                    base_color = SEMANTIC_COLORS.get(sem_class, [255, 255, 255])
                    
                    if is_gt:
                        color_str = f'rgb({base_color[0]},{base_color[1]},{base_color[2]})'
                    else:
                        # Légère variation de couleur pour distinguer les instances prédites de la même classe
                        np.random.seed(uid)
                        variation = np.random.randint(-30, 30, size=3)
                        c = np.clip(np.array(base_color) + variation, 0, 255)
                        color_str = f'rgb({c[0]},{c[1]},{c[2]})'
                        
                    fig.add_trace(go.Scatter3d(
                        x=X_frame[uid_mask, 0], y=X_frame[uid_mask, 1], z=X_frame[uid_mask, 2],
                        mode='markers',
                        marker=dict(size=3, color=color_str, opacity=1.0, line=dict(width=0)),
                        name=f'{class_name} {uid}',
                        text=[f"ID: {uid}<br>Class: {class_name} ({sem_class})"] * np.sum(uid_mask),
                        hoverinfo='text'
                    ), row=1, col=col)
                    
        add_frame_traces(1, GT_frame, is_gt=True)
        add_frame_traces(2, labels_frame, is_gt=False)
        
        scene_config = dict(
            aspectmode='data',
            xaxis_title='X Monde (m)', yaxis_title='Y Monde (m)', zaxis_title='Z Monde (m)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            bgcolor='rgb(20, 24, 34)', # Fond sombre élégant
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title='')
        )
        
        fig.update_layout(title_text=f"<b>Frame {FRAME_TO_SHOW}</b> - Segmentation Panoptique 3D", 
                          title_font=dict(color='white', size=20),
                          paper_bgcolor='rgb(20, 24, 34)',
                          plot_bgcolor='rgb(20, 24, 34)',
                          height=800, showlegend=False,
                          scene=scene_config, scene2=scene_config,
                          margin=dict(l=0, r=0, b=0, t=50))
        
        # Subplot titles formatting
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=16, color='white')
            
        pass # fig.show()
    else:
        print(f"La frame {FRAME_TO_SHOW} n'est pas présente dans le sous-ensemble actuel.")
else:
    print("Pas de données à afficher.")


# @title 6.2 Visualisation Interactive 4D Panoptic & Tracking (SemanticKITTI BEV Clone)
try:
    from google.colab import output
    output.enable_custom_widget_manager()
except ImportError:
    pass

import os
import glob
import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

DOWNSAMPLE_FACTOR = 1

def get_obb_corners(cx, cy, cz, l, w, h, yaw):
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    R = np.array([
        [cos_y, -sin_y, 0],
        [sin_y,  cos_y, 0],
        [    0,      0, 1]
    ])
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    corners = np.vstack([x_corners, y_corners, z_corners])
    corners_3d = np.dot(R, corners)
    corners_3d[0, :] += cx
    corners_3d[1, :] += cy
    corners_3d[2, :] += cz
    return corners_3d

def compute_obb(pts):
    if len(pts) < 3: return None
    xy = pts[:, :2]
    cov = np.cov(xy.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    major_axis = eigenvectors[:, np.argmax(eigenvalues)]
    yaw = np.arctan2(major_axis[1], major_axis[0])
    cos_t, sin_t = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    aligned_xy = xy @ R.T
    min_x, max_x = np.min(aligned_xy[:, 0]), np.max(aligned_xy[:, 0])
    min_y, max_y = np.min(aligned_xy[:, 1]), np.max(aligned_xy[:, 1])
    dim_x = max_x - min_x
    dim_y = max_y - min_y
    center_aligned_xy = np.array([min_x + dim_x / 2, min_y + dim_y / 2])
    c_xy = center_aligned_xy @ np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    dim_z = np.max(pts[:, 2]) - np.min(pts[:, 2])
    cz = np.min(pts[:, 2]) + dim_z / 2
    return c_xy[0], c_xy[1], cz, dim_x, dim_y, dim_z, yaw

def create_obb_lines(obbs):
    x_lines, y_lines, z_lines = [], [], []
    lines_idx = [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]]
    for obb in obbs:
        corners = get_obb_corners(*obb)
        for idx in lines_idx:
            x_lines.extend([corners[0, idx[0]], corners[0, idx[1]], None])
            y_lines.extend([corners[1, idx[0]], corners[1, idx[1]], None])
            z_lines.extend([corners[2, idx[0]], corners[2, idx[1]], None])
    return x_lines, y_lines, z_lines

seq_str = f"{SEQUENCE_TO_TEST:02d}" if 'SEQUENCE_TO_TEST' in globals() else "08"
velo_path = loader.velo_path if 'loader' in globals() else os.path.join("semantic_kitti_data", seq_str, "velodyne")
pred_path = pred_labels_dir if 'pred_labels_dir' in globals() else os.path.join("eval_predictions", "sequences", seq_str, "predictions")

velo_files = sorted(glob.glob(os.path.join(velo_path, "*.bin")))
pred_files = sorted(glob.glob(os.path.join(pred_path, "*.label")))

if 'NUM_FRAMES' in globals() and NUM_FRAMES != -1:
    start = globals().get('START_FRAME', 0)
    velo_files = velo_files[start:start+NUM_FRAMES]
    pred_files = pred_files[:NUM_FRAMES]

def load_frame_data(f_idx):
    if f_idx >= len(velo_files): return None, None, None, None, None
    scan = np.fromfile(velo_files[f_idx], dtype=np.float32).reshape(-1, 4)
    points = scan[:, :3]
    if f_idx < len(pred_files):
        labels = np.fromfile(pred_files[f_idx], dtype=np.uint32)
        inst_labels = labels >> 16
        sem_labels = labels & 0xFFFF
    else:
        inst_labels = np.zeros(len(points), dtype=np.uint32)
        sem_labels = np.zeros(len(points), dtype=np.uint32)
    return points[:, 0], points[:, 1], points[:, 2], inst_labels, sem_labels

cmap = plt.get_cmap('tab20')
colors = [mcolors.to_hex(cmap(i)) for i in range(20)]

x_pt, y_pt, z_pt, inst, sem = load_frame_data(0)
obbs = []
if x_pt is not None:
    idx = np.random.choice(len(x_pt), len(x_pt) // DOWNSAMPLE_FACTOR, replace=False)
    c_pt = np.array(['#A9A9A9'] * len(idx))
    inst_idx = inst[idx]
    unique_inst = np.unique(inst[inst > 0])
    for inst_id in unique_inst:
        mask_all = (inst == inst_id)
        pts = np.column_stack((x_pt[mask_all], y_pt[mask_all], z_pt[mask_all]))
        obb = compute_obb(pts)
        if obb: obbs.append(obb)
    for i, inst_id in enumerate(np.unique(inst_idx[inst_idx > 0])):
        mask = (inst_idx == inst_id)
        c_pt[mask] = colors[inst_id % 20]
else:
    x_pt, y_pt, z_pt, c_pt, idx = np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array(['#000000']), np.array([0])

cx, cy, cz = create_obb_lines(obbs)

scatter_cloud = go.Scatter3d(
    x=x_pt[idx] if x_pt is not None else [], y=y_pt[idx] if y_pt is not None else [], z=z_pt[idx] if z_pt is not None else [],
    mode='markers',
    marker=dict(size=1.5, color=c_pt, line=dict(width=0)),
    name="Predictions 4D"
)
scatter_obbs = go.Scatter3d(x=cx, y=cy, z=cz, mode='lines', line=dict(color='midnightblue', width=4), opacity=1.0, name="OBB Tracks")

scene_config = dict(
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showbackground=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showbackground=False, visible=False),
    zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, showbackground=False, visible=False),
    camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=0, z=2.5)),
    aspectmode='data'
)

fig = go.FigureWidget(data=[scatter_cloud, scatter_obbs])
fig.update_layout(
    scene=scene_config,
    paper_bgcolor='white',
    plot_bgcolor='white',
    margin=dict(l=0, r=0, b=0, t=0),
    showlegend=True,
    height=800
)

slider = widgets.IntSlider(min=0, max=max(0, len(velo_files)-1), step=1, value=0, description='Frame:')

def update_frame(change):
    f_idx = change.new
    x_pt, y_pt, z_pt, inst, sem = load_frame_data(f_idx)
    if x_pt is not None:
        idx = np.random.choice(len(x_pt), len(x_pt) // DOWNSAMPLE_FACTOR, replace=False)
        c_pt = np.array(['#A9A9A9'] * len(idx))
        inst_idx = inst[idx]
        obbs = []
        unique_inst = np.unique(inst[inst > 0])
        for inst_id in unique_inst:
            mask_all = (inst == inst_id)
            pts = np.column_stack((x_pt[mask_all], y_pt[mask_all], z_pt[mask_all]))
            obb = compute_obb(pts)
            if obb: obbs.append(obb)
        for i, inst_id in enumerate(np.unique(inst_idx[inst_idx > 0])):
            mask = (inst_idx == inst_id)
            c_pt[mask] = colors[inst_id % 20]
        
        cx, cy, cz = create_obb_lines(obbs)
        
        with fig.batch_update():
            fig.data[0].x = x_pt[idx]
            fig.data[0].y = y_pt[idx]
            fig.data[0].z = z_pt[idx]
            fig.data[0].marker.color = c_pt
            fig.data[1].x = cx
            fig.data[1].y = cy
            fig.data[1].z = cz

slider.observe(update_frame, names='value')
display(widgets.VBox([slider, fig]))


# @title 6.2 [DEBUG] Visualisation des Scores d'Association (Matrice UOT)
import plotly.express as px
import pandas as pd

if TRACKING_VERBOSE and 'x_matrix_conf' in locals() and len(confirmed_tracks) > 0:
    fig_m = px.imshow(x_matrix_conf.T, 
                    labels=dict(x="Points du Nuage", y="Tracks Confirmées (M)", color="Poids UOT"),
                    title="Matrice de Transport UOT Non-Normalisé",
                    color_continuous_scale='Viridis_r')
    fig_m.show()
else:
    print("Pas de matrice UOT Confirmed à afficher pour cette dernière itération.")


# @title 6.3 [DEBUG] Profiling des Vitesses Estimées (Kalman)
import plotly.graph_objects as go

if 'tracks_dict' in locals() and len(tracks_dict) > 0:
    for tid, tr in tracks_dict.items():
        v_norm = np.linalg.norm(tr.velocity)
        print(f"Track {tid} (Class {tr.semantic_class}, {tr.state}): Vitesse = {v_norm:.2f} m/s")
else:
    print("Aucune piste active à analyser.")

