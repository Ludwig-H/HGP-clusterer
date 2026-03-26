import json

file_path = "tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb"
with open(file_path, "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and cell["source"] and "4.1 HGP Clustering" in cell["source"][0]:
        new_source = []
        for line in cell["source"]:
            # Update the purity check
            if 'if gini_p < 1e-6: return False # Déjà pur' in line:
                new_source.append('    if gini_p < 1e-3: return False # Déjà très pur\n')
            # Update the split condition
            elif 'return gini_c_weighted < (gini_p - 1e-6)' in line:
                new_source.append('    # On split si l\'impureté diminue de façon notable\n')
                new_source.append('    return gini_c_weighted < (gini_p - 1e-3)\n')
            else:
                new_source.append(line)
        cell["source"] = new_source

with open(file_path, "w") as f:
    json.dump(nb, f, indent=2)

print("Epsilon updated to 1e-3 in oracle_Gini.")
