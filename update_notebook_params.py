import json

with open("tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        # Modification de la section 4.1 HGP Clustering
        if cell["source"] and "4.1 HGP Clustering" in cell["source"][0]:
            new_source = []
            for line in cell["source"]:
                # 1. Mettre MIN_CLUSTER_SIZE à 1
                if 'MIN_CLUSTER_SIZE = 50' in line:
                    new_source.append('MIN_CLUSTER_SIZE = 1 # @param {type:"integer"}\n')
                
                # 2. Remplacer DT_FRAME par un calcul basé sur DT_SCALE et un facteur
                elif 'DT_FRAME = 0.25' in line:
                    new_source.append('DBSCAN_FACTOR = 2.0 # @param {type:"number"}\n')
                    new_source.append('DT_FRAME = DBSCAN_FACTOR * DT_SCALE\n')
                
                else:
                    new_source.append(line)
            cell["source"] = new_source

with open("tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("Notebook parameters updated: MIN_CLUSTER_SIZE=1 and DT_FRAME=DBSCAN_FACTOR*DT_SCALE.")
