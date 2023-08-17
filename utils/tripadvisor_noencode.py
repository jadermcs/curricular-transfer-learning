import json
import string
import random
import pathlib
import numpy as np
from tqdm import tqdm
from .nlp import normalize_trip

random.seed(43)
MSG_MAX_SIZE = 1000
MSG_MIN_SIZE = 30

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
                if len(utt["utterance"]) < MSG_MIN_SIZE:
                    continue
                if utt["utterance"].startswith("This topic has been closed"):
                    continue
                encode["text"] += " "+normalize_trip(utt["utterance"][:MSG_MAX_SIZE].lower())
            if i % 9 != 0: ftrain.writelines(json.dumps(encode) + "\n")
            else:          fvalid.writelines(json.dumps(encode) + "\n")

if __name__ == "__main__":
    main()
