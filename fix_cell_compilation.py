import json
import glob

notebooks = [
    "tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb",
    "tests/SemanticKITTI/HGP-Clusterer_SemanticKITTI_3D_then_4D_Panoptic.ipynb"
]

for file_path in notebooks:
    with open(file_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    updated = False
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code" and cell["source"]:
            # Check cell 2 (index 1 conceptually in some counts, let's just check content)
            source_text = "".join(cell["source"])
            if "if BACKEND == 'cgal':\n" in source_text and "    !apt-get install" in source_text:
                # The issue is that python in a normal cell doesn't understand ! syntax properly if mixed with python control flow unless it's handled by IPython magic.
                # Actually, in Colab, mixing python `if` and `!apt-get` works IF correctly indented and IF it's executed by IPython.
                # However, the previous sed might have messed up the indents or we should check if `import` is needed.
                pass # The error from my test script was because `compile()` doesn't handle `!` bash magic. Colab does.
    
    # Let me check where the actual error reported by the user could be.
    # The user said "La cellule ne compile plus..." after I modified DBSCAN_FACTOR
    
print("Done.")
