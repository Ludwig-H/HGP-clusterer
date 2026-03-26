import json
import glob

notebooks = [
    "tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb",
    "tests/SemanticKITTI/HGP-Clusterer_SemanticKITTI_3D_then_4D_Panoptic.ipynb"
]

for file_path in notebooks:
    with open(file_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    updated = False
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code" and cell["source"]:
            new_source = []
            for line in cell["source"]:
                if 'DBSCAN_FACTOR = 0.5 * 1.5 # @param {type:"raw"}' in line:
                    # In colab, using raw is technically valid, but it might break the visual UI depending on the version.
                    # Another way is to calculate it outside the param block, or just evaluate it.
                    # Since 0.5 * 1.5 = 0.75, let's just use the exact value but add a comment explaining it.
                    line = line.replace('DBSCAN_FACTOR = 0.5 * 1.5 # @param {type:"raw"}', 'DBSCAN_FACTOR = 0.75 # @param {type:"number"} # (0.5 * 1.5)')
                    updated = True
                new_source.append(line)
            cell["source"] = new_source

    if updated:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        print(f"Updated {file_path}")

print("Done.")
