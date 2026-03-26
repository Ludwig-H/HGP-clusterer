import json

with open("tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        if cell["source"] and "3.1 Construction du Nuage 4D" in cell["source"][0]:
            new_source = []
            for line in cell["source"]:
                if "NUM_FRAMES = 10" in line:
                    new_source.append('NUM_FRAMES = 10 # @param {type:"integer"} (-1 pour toutes les frames)\n')
                elif "print(f\"Chargement frames {START_FRAME} -> {START_FRAME + NUM_FRAMES}...\")" in line:
                    new_source.append('    actual_num_frames = len(loader) - START_FRAME if NUM_FRAMES == -1 else NUM_FRAMES\n')
                    new_source.append('    print(f"Chargement frames {START_FRAME} -> {START_FRAME + actual_num_frames}...")\n')
                elif "for i in range(NUM_FRAMES):" in line:
                    new_source.append('    for i in range(actual_num_frames):\n')
                elif "t_syn = np.random.randint(0, NUM_FRAMES, size=n_samples) * DT_SCALE" in line:
                    new_source.append('    synth_frames = NUM_FRAMES if NUM_FRAMES != -1 else 10\n')
                    new_source.append('    t_syn = np.random.randint(0, synth_frames, size=n_samples) * DT_SCALE\n')
                else:
                    new_source.append(line)
            cell["source"] = new_source

        elif cell["source"] and any("5.1" in line for line in cell["source"]):
            new_source = []
            for line in cell["source"]:
                if "for i in range(NUM_FRAMES):" in line:
                    new_source.append('    actual_num_frames = len(loader) - START_FRAME if NUM_FRAMES == -1 else NUM_FRAMES\n')
                    new_source.append('    for i in range(actual_num_frames):\n')
                else:
                    new_source.append(line)
            cell["source"] = new_source

with open("tests/SemanticKITTI/HGP_SemanticKITTI_4D_Panoptic.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("Notebook updated for NUM_FRAMES.")
