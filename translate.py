def translate(line: str):
    subwords = line.replace("▁", " ")[1:]  # ▁ is not same _
    return subwords

with open("predictions.model_best.txt", "r", encoding="utf-8") as input_file, \
     open("predictions.model_best.translate.txt", "w", encoding="utf-8") as output_file:
    for line in input_file:
        transformed_line = translate(line.strip())
        output_file.write(transformed_line + "\n")
