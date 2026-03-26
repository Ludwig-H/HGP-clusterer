import json

file_path = "tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb"
with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code" and cell["source"] and any("evaluate_panoptic.py" in line for line in cell["source"]):
        new_source = []
        for line in cell["source"]:
            # On change le split de 'custom' vers 'valid' pour l'injection YAML
            if "cfg['split']['custom'] = [SEQUENCE_TO_TEST]" in line:
                new_source.append("    cfg['split']['valid'] = [SEQUENCE_TO_TEST]\n")
            # On change l'appel shell pour utiliser --split valid
            elif "--split custom" in line:
                new_source.append(line.replace("--split custom", "--split valid"))
            else:
                new_source.append(line)
        cell["source"] = new_source

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Evaluation split fixed to 'valid' with YAML injection.")
