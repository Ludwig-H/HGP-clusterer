import json
import re

with open('tests/SemanticKITTI/HGP-Clusterer_SemanticKITTI_3D_then_4D_Panoptic.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if cell['source'] and '4.1 HGP Clustering' in cell['source'][0]:
            # Nettoyer les doublons de commentaires sur les priors
            new_source = []
            skip_next = False
            for line in cell['source']:
                if skip_next:
                    if '# Alpine utilise la plus petite dimension' in line:
                        skip_next = False
                        continue
                if "# --- Priors de tailles d'objets inspirés de Alpine" in line:
                    skip_next = True
                    continue
                if 'SPLITTING_REGISTRY = {' in line:
                    new_source.append(line)
                    new_source.append('}\n')
                    continue
                new_source.append(line)
            cell['source'] = new_source

with open('tests/SemanticKITTI/HGP-Clusterer_SemanticKITTI_3D_then_4D_Panoptic.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)
