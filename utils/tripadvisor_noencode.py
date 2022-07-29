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

    data = []
    
    (path/"train").mkdir(exist_ok=True)
    trainpath = path/"train/noencoded.json"
    (path/"valid").mkdir(exist_ok=True)
    validpath = path/"valid/noencoded.json"

    with (path/"dialogues.jl").open() as fin:
        for line in fin.readlines():
            data.append(json.loads(line))
    random.shuffle(data)
    with trainpath.open("w") as ftrain, validpath.open("w") as fvalid:
        for i, dialogue in enumerate(tqdm(data)):
            encode = {"id": f"{i}", "text": ""}
            for utt in dialogue["utterances"][:50*2]:
                encode["text"] += " "+normalize(utt["utterance"][:MSG_MAX_SIZE].lower())
            if i % 3 != 0: ftrain.writelines(json.dumps(encode) + "\n")
            else:          fvalid.writelines(json.dumps(encode) + "\n")

if __name__ == "__main__":
    main()
