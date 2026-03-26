import json

file_path = "tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb"
with open(file_path, "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and cell["source"] and "4.1 HGP Clustering" in cell["source"][0]:
        new_source = []
        for line in cell["source"]:
            # Add EPS_GINI parameter
            if 'DT_FRAME = DBSCAN_FACTOR * DT_SCALE' in line:
                new_source.append(line)
                new_source.append('EPS_GINI = 0.001 # @param {type:"number"}\n')
            # Replace 1e-3 with EPS_GINI
            elif 'if gini_p < 1e-3: return False' in line:
                new_source.append('    if gini_p < EPS_GINI: return False # Déjà très pur\n')
            elif 'return gini_c_weighted < (gini_p - 1e-3)' in line:
                new_source.append('    return gini_c_weighted < (gini_p - EPS_GINI)\n')
            else:
                new_source.append(line)
        cell["source"] = new_source

with open(file_path, "w") as f:
    json.dump(nb, f, indent=2)

print("Renamed epsilon to EPS_GINI in oracle_Gini.")
