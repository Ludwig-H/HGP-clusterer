import json

with open("tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb", "r") as f:
    nb = json.load(f)

new_source = """# @title 5.1 Évaluation Officielle SemanticKITTI (PQ, SQ, RQ)
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
    cfg['split']['custom'] = [SEQUENCE_TO_TEST]
    with open(custom_cfg_path, 'w') as f:
        yaml.dump(cfg, f)
    
    # On reconstruit les labels prédits frame par frame
    for i in range(NUM_FRAMES):
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
        
    print("Fichiers de prédiction générés. Lancement de evaluate_panoptic.py...")
    # On lance le script officiel sur ce mini-dataset de test
    !python /content/semantic-kitti-api/evaluate_panoptic.py --dataset {eval_dir} --predictions {pred_dir} --split custom --data_cfg {custom_cfg_path}
else:
    print("Pas de données pour l'évaluation.")
"""

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        # Check if the title starts with "5.1"
        if cell["source"] and any("5.1" in line for line in cell["source"]):
            cell["source"] = [line + "\n" for line in new_source.split("\n")][:-1]
            break

with open("tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("Notebook updated.")
