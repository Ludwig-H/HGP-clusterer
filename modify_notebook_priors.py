import json

with open('tests/SemanticKITTI/HGP-Clusterer_SemanticKITTI_3D_then_4D_Panoptic.ipynb', 'r') as f:
    nb = json.load(f)

# Find the clustering cell
for i, cell in enumerate(nb['cells']):
    if len(cell['source']) > 0 and '4.1 HGP Clustering' in cell['source'][0]:
        new_source = []
        for line in cell['source']:
            if 'DBSCAN_FACTOR =' in line or 'DT_FRAME =' in line:
                continue # remove these lines
            new_source.append(line)
        
        # Insert the priors before the frame loop
        insert_idx = -1
        for j, line in enumerate(new_source):
            if 'frames = np.unique(Time_idx)' in line:
                insert_idx = j
                break
                
        priors_code = [
            "# --- Priors de tailles d'objets inspirés de Alpine (Valeo.ai) ---\n",
            "# Les valeurs sont des Bounding Boxes 2D (Length, Width) en mètres.\n",
            "# Alpine utilise la plus petite dimension (Width) comme seuil pour le clustering.\n",
            "ALPINE_PRIORS = {\n",
            "    1: {\"name\": \"car\", \"L\": 4.4, \"W\": 1.8},\n",
            "    2: {\"name\": \"bicycle\", \"L\": 1.75, \"W\": 0.61},\n",
            "    3: {\"name\": \"motorcycle\", \"L\": 2.2, \"W\": 0.95},\n",
            "    4: {\"name\": \"truck\", \"L\": 10.0, \"W\": 3.0},\n",
            "    5: {\"name\": \"other-vehicle\", \"L\": 10.0, \"W\": 3.0},\n",
            "    6: {\"name\": \"person\", \"L\": 0.94, \"W\": 0.94},\n",
            "    7: {\"name\": \"bicyclist\", \"L\": 1.75, \"W\": 0.61},\n",
            "    8: {\"name\": \"motorcyclist\", \"L\": 2.2, \"W\": 0.95},\n",
            "}\n",
            "\n"
        ]
        
        new_source = new_source[:insert_idx] + priors_code + new_source[insert_idx:]
        
        # Replace method parameter inside HGPClusterer call
        for j, line in enumerate(new_source):
            if 'method=DT_FRAME' in line:
                new_source[j] = "                method=class_threshold, # Seuil dynamique selon la classe\n"
                # Need to define class_threshold before this line
                
        # Find where to define class_threshold
        for j, line in enumerate(new_source):
            if 'clusterer = HGPClusterer(' in line:
                class_thresh_code = [
                    "            # Alpine utilise la largeur (Width) comme seuil de distance\n",
                    "            prior = ALPINE_PRIORS.get(semantic_class, {\"W\": 1.0})\n",
                    "            class_threshold = prior[\"W\"]\n",
                    "            print(f\"  -> Seuil HGP (Width prior) : {class_threshold}m\")\n"
                ]
                new_source = new_source[:j] + class_thresh_code + new_source[j:]
                break
                
        cell['source'] = new_source

with open('tests/SemanticKITTI/HGP-Clusterer_SemanticKITTI_3D_then_4D_Panoptic.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

