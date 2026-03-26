import json
import glob

notebooks = [
    "tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb",
    "tests/SemanticKITTI/HGP-Clusterer_SemanticKITTI_3D_then_4D_Panoptic.ipynb"
]

for file_path in notebooks:
    with open(file_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and cell["source"]:
            new_source = []
            for line in cell["source"]:
                if '!python /content/semantic-kitti-api/evaluate_panoptic.py' in line and '--output' not in line:
                    line = line.rstrip() + ' --output /content/eval_output\n'
                    new_source.append(line)
                    # Add code to read the output
                    new_source.append('    if os.path.exists("/content/eval_output/scores.txt"):\n')
                    new_source.append('        print("\\n--- RÉSULTATS DE L\'ÉVALUATION ---")\n')
                    new_source.append('        with open("/content/eval_output/scores.txt", "r") as f:\n')
                    new_source.append('            print(f.read())\n')
                else:
                    new_source.append(line)
            cell["source"] = new_source

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)

print("Updated notebooks for evaluation output.")
