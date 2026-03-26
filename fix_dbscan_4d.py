import json

file_path = "tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb"
with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

updated = False
for cell in nb.get("cells", []):
    if cell["cell_type"] == "code" and cell["source"]:
        new_source = []
        for line in cell["source"]:
            if 'DBSCAN_FACTOR =' in line and '0.5' in line and '1.5' not in line:
                line = line.replace('0.5', '0.5 * 1.5')
                updated = True
            new_source.append(line)
        cell["source"] = new_source

if updated:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Updated DBSCAN_FACTOR in {file_path}")
