import json
import string
import random
import pathlib
import numpy as np
from tqdm import tqdm

random.seed(43)
MSG_MAX_SIZE = 1000

def normalize(msg):
    rm_spec = [x for x in msg if x in string.printable]
    return " ".join(rm_spec)

def main():
    path = pathlib.Path("data/tripadvisor/")
    # nlp = spacy.load("en_core_web_sm")
    schema = path/"schema.txt"
    schema.touch(exist_ok=True)
    # with schema.open("w") as fin:
    #     fin.write("ORG")

    data = []
    
    (path/"train").mkdir(exist_ok=True)
    trainpath = path/"train/encoded.json"
    (path/"valid").mkdir(exist_ok=True)
    validpath = path/"valid/encoded.json"

    with (path/"dialogues.jl").open() as fin:
        for line in fin.readlines():
            data.append(json.loads(line))
    random.shuffle(data)
    sizes = []
    with trainpath.open("w") as ftrain, validpath.open("w") as fvalid:
        for i, dialogue in enumerate(tqdm(data)):
            main = normalize(dialogue["utterances"][0]["utterance"][:MSG_MAX_SIZE]).lower()
            for turn, utt in enumerate(dialogue["utterances"][1:50]):
                encode = {"id": f"{i}-{turn}", "text": ""}
                encode["text"] += "<sos_u>"+main+"<eos_u>"
                encode["text"] += "<sos_b>"+dialogue["domain"].lower()+"<eos_b>"
                encode["text"] += "<sos_a> <eos_a>"
                encode["text"] += "<sos_r>"+normalize(utt["utterance"][:MSG_MAX_SIZE]).lower()+"<eos_r>"
                size = len(encode["text"])
                sizes.append(len(encode["text"]))
                if i % 9 != 0: ftrain.writelines(json.dumps(encode) + "\n")
                else:          fvalid.writelines(json.dumps(encode) + "\n")
    print("Min:", np.min(sizes))
    print("Max:", np.max(sizes))
    print("Mean:", np.mean(sizes))
    print("Std:", np.std(sizes))

if __name__ == "__main__":
    main()