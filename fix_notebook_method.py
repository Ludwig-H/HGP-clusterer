import json

with open("tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        if cell["source"] and "4.1 HGP Clustering" in cell["source"][0]:
            new_source = []
            for line in cell["source"]:
                if "cluster_selection_method='dbscan'," in line:
                    # Remove this line, it's not a valid argument
                    continue
                elif "cluster_selection_epsilon=DT_FRAME," in line:
                    # Replace with 'method=DT_FRAME'
                    new_source.append(f"        method=DT_FRAME,\n")
                else:
                    new_source.append(line)
            cell["source"] = new_source

with open("tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("Notebook corrected: method=DT_FRAME used for DBSCAN mode.")
