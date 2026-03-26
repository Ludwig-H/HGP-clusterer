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
                if 'SPLITTING_MODE = "None" # @param ["None", "oracle_Gini"] {allow-input: true}' in line or 'SPLITTING_MODE = "oracle_Gini" # @param ["None", "oracle_Gini"] {allow-input: true}' in line:
                    new_source.append(line)
                    # Insert the comment immediately after SPLITTING_MODE but before DBSCAN_FACTOR
                    new_source.append('# On utilise un facteur sur la dimension W (diamètre) pour obtenir un rayon (1/2).\n')
                    new_source.append('# On peut ajouter un multiplicateur supplémentaire (ex: 1.5) : DBSCAN_FACTOR = 0.5 * 1.5 = 0.75\n')
                    continue
                    
                if 'DBSCAN_FACTOR = 0.75 # @param {type:"number"}' in line:
                    new_source.append(line)
                    updated = True
                    continue
                
                new_source.append(line)
            cell["source"] = new_source

    if updated:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        print(f"Updated {file_path}")

print("Done.")
