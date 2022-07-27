import json
from nlp import normalize
from pathlib import Path
from tqdm import tqdm

def generate_encoded():
    path = Path("data/multiwoz/")
    with (path/"dialog_acts.json").open() as fin:
        acts = json.load(fin)
    progressbar = tqdm(list(path.glob("**/dialogues_*")))

    for split in ["train", "test", "dev"]:
        out_json = {}
        fout = (path/f"{split}/encoded.json").open("w")
        for fname in path.glob(split+"/dialogues_*"):
            progressbar.update(1)
            with fname.open() as data:
                jsondata = json.load(data)
            for dialog in jsondata:
                encoded_string = ""
                services = dialog["services"]
                for turn in dialog["turns"]:
                    if int(turn["turn_id"]) % 2 == 0:
                        utt = normalize(turn["utterance"])
                        encoded_string += "<sos_u> " + utt + " <eos_u>"
                        encoded_string += " <sos_b> "
                        for frame in turn["frames"]:
                            if frame["service"] in services and frame["state"]["active_intent"] != "NONE":
                                encoded_string += "["+frame["state"]["active_intent"]+"] "
                                for value in frame["state"]["requested_slots"]:
                                    encoded_string += value + " "
                                for key, value in frame["state"]["slot_values"].items():
                                    encoded_string += "#"+key.split("-")[-1].lstrip("book") + " " + value[0].lower() + " "
                        encoded_string += "<eos_b> "
                    else:
                        dialog_acts = acts[dialog["dialogue_id"]][turn["turn_id"]]
                        act_string = []
                        response = turn["utterance"]
                        for key in dialog_acts["dialog_act"]:
                            act_string.append("["+key.lower()+"]")
                            for param in dialog_acts["dialog_act"][key]:
                                if param[0] != "none":
                                    param[0] = param[0].lstrip("book")
                                    act_string.append(" ".join(param))
                                    if param[1] != "?":
                                        response = response.replace(param[1], f"[value_{param[0].replace('arriveby', 'arrive')}]")
                        encoded_string += "<sos_a> " + " ".join(act_string) + " <eos_a> "
                        encoded_string += "<sos_r>" + response.lower() + "<eos_r>"
                fout.write(json.dumps({"id": dialog["dialogue_id"], "domain": services,
                "text": encoded_string})+"\n")

if __name__ == "__main__":
    generate_encoded()