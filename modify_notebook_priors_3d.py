import json

with open('tests/SemanticKITTI/HGP-Clusterer_SemanticKITTI_3D_then_4D_Panoptic.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if len(cell['source']) > 0 and '4.1 HGP Clustering' in cell['source'][0]:
        new_source = []
        for line in cell['source']:
            # Mettre le splitting à "None" par défaut
            if 'SPLITTING_MODE = "oracle_Gini"' in line:
                new_source.append('SPLITTING_MODE = "None" # @param ["None", "oracle_Gini"] {allow-input: true}\n')
                new_source.append('DBSCAN_FACTOR = 0.5 # @param {type:"number"}\n') # Ajout du FACTOR_DBSCAN
            elif 'ALPINE_PRIORS = {' in line:
                # On va remplacer tout le bloc ALPINE_PRIORS
                pass
            elif '1: {"name": "car"' in line or '2: {"name": "bicycle"' in line or '3: {"name": "motorcycle"' in line or '4: {"name": "truck"' in line or '5: {"name": "other-vehicle"' in line or '6: {"name": "person"' in line or '7: {"name": "bicyclist"' in line or '8: {"name": "motorcyclist"' in line or (line.strip() == '}' and 'ALPINE_PRIORS' not in ''.join(new_source)):
                pass # Skip the old priors definition
            elif 'class_threshold = prior["W"]' in line:
                new_source.append('            class_threshold = prior["W"] * DBSCAN_FACTOR\n')
            else:
                new_source.append(line)
        
        # Ré-insertion des priors avec la dimension H (Hauteur)
        insert_idx = -1
        for j, line in enumerate(new_source):
            if 'frames = np.unique(Time_idx)' in line:
                insert_idx = j
                break
                
        priors_code = [
            "# --- Priors de tailles d'objets (Inspirés d'Alpine + 3D) ---\n",
            "# Bounding Boxes 3D (Length, Width, Height) en mètres.\n",
            "ALPINE_PRIORS = {\n",
            "    1: {\"name\": \"car\", \"L\": 4.4, \"W\": 1.8, \"H\": 1.5},\n",
            "    2: {\"name\": \"bicycle\", \"L\": 1.75, \"W\": 0.61, \"H\": 1.1},\n",
            "    3: {\"name\": \"motorcycle\", \"L\": 2.2, \"W\": 0.95, \"H\": 1.2},\n",
            "    4: {\"name\": \"truck\", \"L\": 10.0, \"W\": 3.0, \"H\": 3.5},\n",
            "    5: {\"name\": \"other-vehicle\", \"L\": 10.0, \"W\": 3.0, \"H\": 3.0},\n",
            "    6: {\"name\": \"person\", \"L\": 0.94, \"W\": 0.94, \"H\": 1.75},\n",
            "    7: {\"name\": \"bicyclist\", \"L\": 1.75, \"W\": 0.61, \"H\": 1.8},\n",
            "    8: {\"name\": \"motorcyclist\", \"L\": 2.2, \"W\": 0.95, \"H\": 1.8},\n",
            "}\n",
            "\n"
        ]
        
        new_source = new_source[:insert_idx] + priors_code + new_source[insert_idx:]
        cell['source'] = new_source

with open('tests/SemanticKITTI/HGP-Clusterer_SemanticKITTI_3D_then_4D_Panoptic.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

