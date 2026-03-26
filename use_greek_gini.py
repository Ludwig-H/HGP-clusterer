import json

file_path = "tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb"
with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and cell["source"] and "4.1 HGP Clustering" in cell["source"][0]:
        new_source = []
        for line in cell["source"]:
            # Replace EPS_GINI with ε_Gini
            new_line = line.replace('EPS_GINI', 'ε_Gini')
            new_source.append(new_line)
        cell["source"] = new_source

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Variable EPS_GINI renamed to ε_Gini in the notebook.")
