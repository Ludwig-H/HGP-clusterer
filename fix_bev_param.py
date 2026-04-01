import json

files = ["tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb", "tests/SemanticKITTI/HGP-Clusterer_SemanticKITTI_3D_then_4D_Panoptic.ipynb"]
for file_path in files:
    with open(file_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    updated = False
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code" and cell["source"]:
            new_source = []
            for line in cell["source"]:
                if 'APPLY_BEV = True # @param {type:"boolean"}' in line:
                    line = line.replace('APPLY_BEV = True', 'APPLY_BEV = False')
                    updated = True
                if 'VERBOSE_DEBUG = False # @param {type:"boolean"}' in line:
                    line = line.replace('VERBOSE_DEBUG = False', 'VERBOSE_DEBUG = True')
                    updated = True
                new_source.append(line)
            cell["source"] = new_source

    if updated:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        print(f"Updated BEV param in {file_path}")

print("Done.")
