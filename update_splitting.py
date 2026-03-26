import json

with open("tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        if cell["source"] and "4.1 HGP Clustering" in cell["source"][0]:
            new_source = []
            found_params = False
            for line in cell["source"]:
                if 'EXP_Z = 1' in line:
                    new_source.append(line)
                    if not found_params:
                        new_source.append('\n')
                        new_source.append('def oracle_Gini(parent_pts_idx, children_pts_idx_list):\n')
                        new_source.append('    """\n')
                        new_source.append('    Splitting basé sur l\'index de Gini (Oracle).\n')
                        new_source.append('    Divise si l\'impureté (Gini) des instances GT est réduite par le split.\n')
                        new_source.append('    """\n')
                        new_source.append('    global Y_inst\n')
                        new_source.append('    if Y_inst is None or len(parent_pts_idx) == 0: return False\n')
                        new_source.append('    \n')
                        new_source.append('    def get_gini(indices):\n')
                        new_source.append('        if len(indices) == 0: return 0.0\n')
                        new_source.append('        labels = Y_inst[indices]\n')
                        new_source.append('        _, counts = np.unique(labels, return_counts=True)\n')
                        new_source.append('        probs = counts / len(indices)\n')
                        new_source.append('        return 1.0 - np.sum(probs**2)\n')
                        new_source.append('    \n')
                        new_source.append('    gini_p = get_gini(parent_pts_idx)\n')
                        new_source.append('    if gini_p == 0: return False # Déjà pur\n')
                        new_source.append('    \n')
                        new_source.append('    n_total = len(parent_pts_idx)\n')
                        new_source.append('    gini_c_weighted = 0.0\n')
                        new_source.append('    for c_pts in children_pts_idx_list:\n')
                        new_source.append('        if len(c_pts) == 0: continue\n')
                        new_source.append('        gini_c_weighted += (len(c_pts) / n_total) * get_gini(c_pts)\n')
                        new_source.append('    \n')
                        new_source.append('    return gini_c_weighted < gini_p\n')
                        new_source.append('\n')
                        new_source.append('SPLITTING_REGISTRY = {\n')
                        new_source.append('    "None": None,\n')
                        new_source.append('    "oracle_Gini": oracle_Gini\n')
                        new_source.append('}\n')
                        new_source.append('\n')
                        new_source.append('# @geogram_install/include/geogram1/geogram/parameterization/mesh_global_param.h\n')
                        new_source.append('SPLITTING_MODE = "oracle_Gini" # @param ["None", "oracle_Gini"] {allow-input: true}\n')
                        found_params = True
                elif "splitting=None," in line:
                    new_source.append('        splitting=SPLITTING_REGISTRY.get(SPLITTING_MODE),\n')
                else:
                    new_source.append(line)
            cell["source"] = new_source

with open("tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("Notebook updated with oracle_Gini splitting.")
