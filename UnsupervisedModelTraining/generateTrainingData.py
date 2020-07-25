import os
import re

files = os.listdir("Train")


with open('train2.txt', 'a') as outfile:
    for f in files:
        fname = open("Train/"+f, encoding="utf-8")
        clean = fname.read().replace("\n", '')
        clean = clean.replace("’", '')
        clean = clean.replace('“', '')
        clean = clean.replace('”', '')
        clean = clean.replace('"', '')
        clean = clean.replace("'", '')
        clean = re.sub(r"[,.;@#?!&$]+\ *", " ", clean)
        outfile.write(clean+"\n\r")
        fname.close()


"""
with open('output.txt', 'r') as outfile:
    for lines in outfile:
        print(lines+"1")

"""
