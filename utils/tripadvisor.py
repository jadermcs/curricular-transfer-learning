import json
import string
import random
import pathlib
import numpy as np
from tqdm import tqdm
from .nlp import normalize_trip

random.seed(43)
MSG_MAX_SIZE = 1000

def main(label=True):
    path = pathlib.Path("data/tripadvisor/")
    # nlp = spacy.load("en_core_web_sm")
    schema = path/"schema.txt"
    schema.touch(exist_ok=True)
    # with schema.open("w") as fin:
    #     fin.write("ORG")

    data = []
    
    (path/"train").mkdir(exist_ok=True)
    trainpath = path/(f"train/encoded{}.json" % ("" if label else "_nolabel"))
    (path/"valid").mkdir(exist_ok=True)
    validpath = path/(f"valid/encoded{}.json" % ("" if label else "_nolabel"))

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
                encode = {
                        "id": f"{i}-{turn}",
                        "url": dialogue["url"],
                        "text": ""
                        }
                encode["text"] += "<sos_u>"+main+"<eos_u>"
                if label: encode["text"] += "<sos_b>"+dialogue["domain"].split()[0].lower()+"<eos_b>"
                else: encode["text"] += "<sos_b> <eos_b>"

                encode["text"] += "<sos_a> <eos_a>"
                encode["text"] += "<sos_r>"+normalize_trip(utt["utterance"][:MSG_MAX_SIZE])+"<eos_r>"
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
