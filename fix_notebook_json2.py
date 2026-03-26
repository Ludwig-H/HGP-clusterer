import json

with open('tests/SemanticKITTI/HGP-Clusterer_SemanticKITTI_3D_then_4D_Panoptic.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if cell['source'] and '4.1 HGP Clustering' in cell['source'][0]:
            new_source = []
            for line in cell['source']:
                if 'SPLITTING_REGISTRY = {' in line:
                    new_source.append("SPLITTING_REGISTRY = {\n")
                    new_source.append("    \"None\": None,\n")
                    new_source.append("    \"oracle_Gini\": oracle_Gini\n")
                    new_source.append("}\n")
                    continue
                if '    "None": None,' in line or '    "oracle_Gini": oracle_Gini' in line or (line.strip() == '}' and 'SPLITTING_REGISTRY' in ''.join(new_source[-4:])):
                    continue
                if '# Les valeurs sont des Bounding Boxes 2D' in line:
                    continue
                if '# Alpine utilise la plus petite dimension' in line:
                    continue
                new_source.append(line)
            cell['source'] = new_source

with open('tests/SemanticKITTI/HGP-Clusterer_SemanticKITTI_3D_then_4D_Panoptic.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

