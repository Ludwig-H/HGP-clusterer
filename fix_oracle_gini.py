import json

with open("tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        if cell["source"] and "4.1 HGP Clustering" in cell["source"][0]:
            new_source = []
            for line in cell["source"]:
                if 'def oracle_Gini(parent_pts_idx, children_pts_idx_list):' in line:
                    new_source.append('CURRENT_GT_INSTANCES = None\n')
                    new_source.append('\n')
                    new_source.append('def oracle_Gini(parent_pts_idx, children_pts_idx_list):\n')
                    new_source.append('    """\n')
                    new_source.append('    Splitting basé sur l\'index de Gini (Oracle).\n')
                    new_source.append('    Utilise CURRENT_GT_INSTANCES qui doit être mis à jour avant le fit.\n')
                    new_source.append('    """\n')
                    new_source.append('    global CURRENT_GT_INSTANCES\n')
                    new_source.append('    if CURRENT_GT_INSTANCES is None or len(parent_pts_idx) == 0: return False\n')
                    new_source.append('    \n')
                    new_source.append('    def get_gini(indices):\n')
                    new_source.append('        if len(indices) == 0: return 0.0\n')
                    new_source.append('        labels = CURRENT_GT_INSTANCES[indices] # Utilisation des labels locaux à la classe\n')
                    new_source.append('        if len(labels) == 0: return 0.0\n')
                    new_source.append('        _, counts = np.unique(labels, return_counts=True)\n')
                    new_source.append('        probs = counts / len(labels)\n')
                    new_source.append('        return 1.0 - np.sum(probs**2)\n')
                    new_source.append('    \n')
                    new_source.append('    gini_p = get_gini(parent_pts_idx)\n')
                    new_source.append('    if gini_p < 1e-6: return False # Déjà pur\n')
                    new_source.append('    \n')
                    new_source.append('    n_total = len(parent_pts_idx)\n')
                    new_source.append('    gini_c_weighted = 0.0\n')
                    new_source.append('    for c_pts in children_pts_idx_list:\n')
                    new_source.append('        if len(c_pts) == 0: continue\n')
                    new_source.append('        gini_c_weighted += (len(c_pts) / n_total) * get_gini(c_pts)\n')
                    new_source.append('    \n')
                    new_source.append('    # On split si l\'impureté diminue de façon notable\n')
                    new_source.append('    return gini_c_weighted < (gini_p - 1e-6)\n')
                    # On saute les lignes de l'ancienne version
                    continue
                
                # Ignorer les lignes de l'ancienne implémentation jusqu'à SPLITTING_REGISTRY
                if 'if Y_inst is None' in line or 'labels = Y_inst' in line or 'gini_p = get_gini' in line or 'gini_c_weighted < gini_p' in line:
                    continue
                if 'Splitting basé sur l\'index de Gini' in line or 'Divise si l\'impureté' in line:
                    continue
                if 'def get_gini' in line or 'probs = counts' in line or 'return 1.0 - np.sum' in line:
                    continue

                # Mise à jour de la boucle de clustering pour définir CURRENT_GT_INSTANCES
                if 'X_class = X_clustering[class_mask]' in line:
                    new_source.append(line)
                    new_source.append('    Y_inst_class = Y_inst[class_mask]\n')
                    new_source.append('    CURRENT_GT_INSTANCES = Y_inst_class # Mise à jour pour oracle_Gini\n')
                elif 'Y_inst_class = Y_inst[class_mask] # Utile si Oracle Split' in line:
                    continue # On l'a déjà ajouté juste au dessus
                
                else:
                    new_source.append(line)
            cell["source"] = new_source

with open("tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("Notebook corrected: oracle_Gini now uses local class labels.")
