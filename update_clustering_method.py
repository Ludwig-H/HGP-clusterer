import json

with open("tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        if cell["source"] and "4.1 HGP Clustering" in cell["source"][0]:
            new_source = []
            for line in cell["source"]:
                if 'EXP_Z = 2 # @param {type:"number"}' in line:
                    new_source.append('EXP_Z = 1 # @param {type:"number"}\n')
                    new_source.append('DT_FRAME = 0.25 # @param {type:"number"}\n')
                elif 'expZ=EXP_Z,' in line:
                    new_source.append(line)
                    new_source.append("        cluster_selection_method='dbscan',\n")
                    new_source.append("        cluster_selection_epsilon=DT_FRAME,\n")
                else:
                    new_source.append(line)
            cell["source"] = new_source

with open("tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("Notebook updated for DBSCAN method.")
