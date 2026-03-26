import json

notebooks = [
    "tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb",
    "tests/SemanticKITTI/HGP-Clusterer_SemanticKITTI_3D_then_4D_Panoptic.ipynb"
]

for file_path in notebooks:
    with open(file_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"File {file_path} loaded correctly.")
