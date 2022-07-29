import pathlib
import random
import json
from utils.nlp import normalize
from tqdm import tqdm

random.seed(43)

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
        for i, dialogue in enumerate(tqdm(data[:100000])):
            encode = {"id": f"{i}", "text": ""}
            for utt in dialogue["utterances"]:
                encode["text"] += " "+normalize(utt["utterance"])
            if i % 3 != 0: ftrain.writelines(json.dumps(encode) + "\n")
            else:          fvalid.writelines(json.dumps(encode) + "\n")

if __name__ == "__main__":
    main()
