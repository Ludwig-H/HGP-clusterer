import json
import glob

notebooks = glob.glob("tests/SemanticKITTI/*.ipynb")

for file_path in notebooks:
    with open(file_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    updated = False
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code" and cell["source"]:
            new_source = []
            for line in cell["source"]:
                if "BACKEND = 'cgal'  # @param ['geogram', 'cgal']" in line:
                    line = line.replace("BACKEND = 'cgal'  # @param ['geogram', 'cgal']", "BACKEND = 'geogram'  # @param ['geogram', 'cgal']")
                    updated = True
                new_source.append(line)
            cell["source"] = new_source

    if updated:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        print(f"Updated backend in {file_path}")

print("Done.")
