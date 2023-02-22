import json
import string
import random
import pathlib
import numpy as np
from tqdm import tqdm
from .nlp import normalize_trip

random.seed(43)
MSG_MAX_SIZE = 1000

def encode(dialogue, utt, label=True):
    encoded = ""
    encode["text"] += "<sos_u>"+dialogue["main"]+"<eos_u>"
    if label: encode += "<sos_b>"+dialogue["domain"].split()[0].lower()+"<eos_b>"
    else: encode += "<sos_b> <eos_b>"
    encode += "<sos_a> <eos_a>"
    encode += "<sos_r>"+normalize_trip(utt["utterance"][:MSG_MAX_SIZE])+"<eos_r>"
    return encoded

def main(label=True):
    path = pathlib.Path("data/tripadvisor/")

    data = []

    (path/"train").mkdir(exist_ok=True)
    trainpath = path/("train/encoded{}".format(".json" if label else "_nolabel.json"))
    (path/"valid").mkdir(exist_ok=True)
    validpath = path/("valid/encoded{}".format(".json" if label else "_nolabel.json"))

    with (path/"dialogues.jl").open() as fin:
        for line in fin.readlines():
            data.append(json.loads(line))
    random.shuffle(data)
    sizes = []
    with trainpath.open("w") as ftrain, validpath.open("w") as fvalid:
        for i, dialogue in enumerate(tqdm(data)):
            main = normalize_trip(dialogue["utterances"][0]["utterance"][:MSG_MAX_SIZE])
            for turn, utt in enumerate(dialogue["utterances"][1:50]):
                if utt["utterance"].startswith("This topic has been closed"):
                    continue
                dialogue["main"] = main
                encode = {
                        "id": f"{i}-{turn}",
                        "url": dialogue["url"],
                        "text": encode(dialogue, utt, label),
                        }
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
