import pathlib
import json
import spacy



def main():
    path = pathlib.Path("data/tripadvisor/")
    # nlp = spacy.load("en_core_web_sm")
    schema = path/"schema.json"
    with schema.open("w") as fin:
        fin.write("ORG")

    data = []
    proc = []
    
    with (path/"cdialogues.jl").open() as fin:
        for line in fin.readlines():
            data.append(json.loads(line))
    for i, dialogue in enumerate(data):
        for turn, utt in enumerate(dialogue["utterances"][1:]):
            encode = {"id": f"{i}-{turn}", "text": ""}
            encode["text"] += "<sos_u>"+dialogue["utterances"][0]["utterance"]+"<eos_u>"
            encode["text"] += "<sos_b>"+" ".join(dialogue["utterances"][0]["utterance"])+"<eos_b>"
            encode["text"] += "<sos_a>"+" ".join(utt["entities"])+"<eos_a>"
            encode["text"] += "<sos_r>"+utt["utterance"]+"<eos_r>"
            proc.append(encode)

    (path/"train").mkdir(exist_ok=True)
    trainpath = path/"train/encoded.json"
    (path/"valid").mkdir(exist_ok=True)
    validpath = path/"valid/encoded.json"
    with trainpath.open("w") as ftrain, validpath.open("w") as fvalid:
        for index, line in enumerate(proc):
            if index % 3 != 0:
                ftrain.writelines(json.dumps(line) + "\n")
            else:
                fvalid.writelines(json.dumps(line) + "\n")

if __name__ == "__main__":
    main()