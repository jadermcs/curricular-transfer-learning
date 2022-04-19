import pathlib
import json

def main():
    path = pathlib.Path("data/tripadvisor/")

    data = []
    proc = []
    
    with (path/"cdialogues.jl").open() as fin:
        for line in fin.readlines():
            data.append(json.loads(line))
    for i, dialogue in enumerate(data):
        encode = {"id": f"{i}", "text": ""}
        for utt in dialogue["utterances"]:
            encode["text"] += " "+utt["utterance"]
        proc.append(encode)

    (path/"train").mkdir(exist_ok=True)
    trainpath = path/"train/noencode.json"
    (path/"valid").mkdir(exist_ok=True)
    validpath = path/"valid/noencode.json"
    with trainpath.open("w") as ftrain, validpath.open("w") as fvalid:
        for index, line in enumerate(proc):
            if index % 3 != 0:
                ftrain.writelines(json.dumps(line) + "\n")
            else:
                fvalid.writelines(json.dumps(line) + "\n")

if __name__ == "__main__":
    main()