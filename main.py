from gensim.models import Word2Vec
import re
import json
import sys
import glob
import os

# parser = argparse.ArgumentParser(description="Text File to Word2Vec Vectors")

# parser.add_argument("input", help="Path to the input text file")
# parser.add_argument("-o", "--output", default="vector.json",
#                     help="Path to the output text file (default: vector.json)")
argsInput = "data/test.txt"
output_text_file = "output.txt"

listOfFiles = []
if os.path.isdir(argsInput):
    listOfFiles = glob.glob(argsInput + '/*.txt')
else:
    listOfFiles.append(argsInput)

final_sentences = []
for file in listOfFiles:
    text = open(file, encoding="utf8",
                errors='ignore').read().lower().replace("\n", " ")
    sentences = re.split("[.?!]", text)
    for sentence in sentences:
        words = re.split(r'\W+', sentence)
        final_sentences.append(words)


model = Word2Vec(final_sentences, size=100, window=5, min_count=5, workers=4)
model.wv.save_word2vec_format(output_text_file, binary=True)
