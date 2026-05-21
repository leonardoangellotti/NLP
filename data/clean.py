with open("eng-ita.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

cleaned_lines = []
for line in lines:
    if "CC" in line:
        line = line.split("CC")[0].rstrip()
    cleaned_lines.append(line)

with open("output.txt", "w", encoding="utf-8") as f:
    for line in cleaned_lines:
        f.write(line + "\n")
