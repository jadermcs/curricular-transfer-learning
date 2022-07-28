import pathlib
import random
import json
from utils.nlp import normalize
from tqdm import tqdm

random.seed(43)

def main():
    path = pathlib.Path("data/tripadvisor/")
    # nlp = spacy.load("en_core_web_sm")
    schema = path/"schema.json"
    schema.touch(exist_ok=True)
    with schema.open("w") as fin:
        fin.write("ORG")

    data = []
    
    (path/"train").mkdir(exist_ok=True)
    trainpath = path/"train/encoded.json"
    (path/"valid").mkdir(exist_ok=True)
    validpath = path/"valid/encoded.json"

    with (path/"dialogues.jl").open() as fin:
        for line in fin.readlines():
            data.append(json.loads(line))
    random.shuffle(data)
    with trainpath.open("w") as ftrain, validpath.open("w") as fvalid:
        for i, dialogue in enumerate(tqdm(data)):
            for turn, utt in enumerate(dialogue["utterances"][1:]):
                encode = {"id": f"{i}-{turn}", "text": ""}
                encode["text"] += "<sos_u>"+normalize(dialogue["utterances"][0]["utterance"])+"<eos_u>"
                encode["text"] += "<sos_b>"+dialogue["domain"].lower()+"<eos_b>"
                encode["text"] += "<sos_a> <eos_a>"
                encode["text"] += "<sos_r>"+normalize(utt["utterance"])+"<eos_r>"
                if i % 3 != 0: ftrain.writelines(json.dumps(encode) + "\n")
                else:          fvalid.writelines(json.dumps(encode) + "\n")

if __name__ == "__main__":
    main()
