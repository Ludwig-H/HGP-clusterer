import json

file_path = "tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb"
with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and cell["source"] and "3.1 Construction du Nuage 4D" in cell["source"][0]:
        new_source = []
        for line in cell["source"]:
            if 'DT_SCALE = 0.25' in line:
                new_source.append(line.replace('0.25', '0.5'))
            else:
                new_source.append(line)
        cell["source"] = new_source

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("DT_SCALE updated to 0.5 in the notebook.")
