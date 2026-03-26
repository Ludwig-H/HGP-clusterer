import json
import glob

notebooks = [
    "tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb",
    "tests/SemanticKITTI/HGP-Clusterer_SemanticKITTI_3D_then_4D_Panoptic.ipynb"
]

NEW_PRIORS = [
    "    1: {\"name\": \"car\",           \"L\": 4.5,  \"W\": 1.85, \"H\": 1.6},\n",
    "    2: {\"name\": \"bicycle\",       \"L\": 1.8,  \"W\": 0.6,  \"H\": 1.1},\n",
    "    3: {\"name\": \"motorcycle\",    \"L\": 2.2,  \"W\": 0.9,  \"H\": 1.3},\n",
    "    4: {\"name\": \"truck\",         \"L\": 10.0, \"W\": 2.6,  \"H\": 3.5},\n",
    "    5: {\"name\": \"other-vehicle\", \"L\": 12.0, \"W\": 2.6,  \"H\": 3.5},\n",
    "    6: {\"name\": \"person\",        \"L\": 0.6,  \"W\": 0.6,  \"H\": 1.75},\n",
    "    7: {\"name\": \"bicyclist\",     \"L\": 1.8,  \"W\": 0.75, \"H\": 1.8},\n",
    "    8: {\"name\": \"motorcyclist\",  \"L\": 2.2,  \"W\": 0.9,  \"H\": 1.8},\n"
]

for file_path in notebooks:
    with open(file_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    updated = False
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code" and cell["source"]:
            new_source = []
            skip_priors = False
            for line in cell["source"]:
                if 'DBSCAN_FACTOR = 0.5' in line and '0.5 * 1.5' not in line:
                    line = line.replace('0.5', '0.5 * 1.5')
                    updated = True
                
                if 'ALPINE_PRIORS = {' in line:
                    skip_priors = True
                    new_source.append(line)
                    new_source.extend(NEW_PRIORS)
                    updated = True
                    continue
                
                if skip_priors:
                    if '}' in line and '1:' not in line and '2:' not in line and '3:' not in line and '4:' not in line and '5:' not in line and '6:' not in line and '7:' not in line and '8:' not in line:
                        skip_priors = False
                        new_source.append(line)
                    continue

                new_source.append(line)
            cell["source"] = new_source

    if updated:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        print(f"Updated {file_path}")

print("Done.")
