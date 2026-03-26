import json

file_path = "tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb"
with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and cell["source"] and "6.1 Visualisation 3D" in cell["source"][0]:
        new_source = []
        for line in cell["source"]:
            if 'GT_v = Y_inst[idx]' in line:
                new_source.append(line)
                new_source.append('    S_v = Y_sem[idx]\n')
                new_source.append('    THINGS_NAMES = {\n')
                new_source.append('        1: "car", 2: "bicycle", 3: "motorcycle", 4: "truck",\n')
                new_source.append('        5: "other-vehicle", 6: "person", 7: "bicyclist", 8: "motorcyclist"\n')
                new_source.append('    }\n')
            elif 'hover_text = [f"GT: {l}<br>Pred: {L_v[m][i]}\" for i in range(np.sum(m))]' in line:
                new_source.append('        # Récupération du nom de la classe\n')
                new_source.append('        class_name = THINGS_NAMES.get(S_v[m][0], "unknown")\n')
                new_source.append('        hover_text = [f"Class: {class_name}<br>GT Instance: {l}<br>Pred Instance: {L_v[m][i]}" for i in range(np.sum(m))]\n')
            elif 'hover_text = [f"Pred: {l}<br>GT: {GT_v[m][i]}\" for i in range(np.sum(m))]' in line:
                new_source.append('        # Récupération du nom de la classe\n')
                new_source.append('        class_name = THINGS_NAMES.get(S_v[m][0], "unknown")\n')
                new_source.append('        hover_text = [f"Class: {class_name}<br>Pred Instance: {l}<br>GT Instance: {GT_v[m][i]}" for i in range(np.sum(m))]\n')
            else:
                new_source.append(line)
        cell["source"] = new_source

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Visualization hover text updated with class names.")
