import json
from .nlp import normalize
from pathlib import Path
from tqdm import tqdm

translate = {"leaveat":"leave", "arriveby":"arrive"}

def parse_state(turn):
        state = {}
        for frame in turn["frames"]:  
            domain = frame["service"]
            domain_state = {}
            slots = frame["state"]["slot_values"]
            for name, value in slots.items():
                if "dontcare" in value:
                    continue 
                domain_state[name.split('-')[1]] = value[0]
            
            if domain_state:
                state[domain] = domain_state
                state[domain]["requested_slots"] = []
                for request in frame["state"]["requested_slots"]:
                    state[domain]["requested_slots"].append(request)
        return state

def generate_encoded():
    path = Path("data/multiwoz/")
    with (path/"dialog_acts.json").open() as fin:
        acts = json.load(fin)
    progressbar = tqdm(list(path.glob("**/dialogues_*")))

    special_tokens = set()
    for split in ["train", "test", "dev"]:
        fout = (path/f"{split}/encoded.json").open("w")
        for fname in path.glob(split+"/dialogues_*"):
            progressbar.update(1)
            with fname.open() as data:
                jsondata = json.load(data)
            for dialog in jsondata:
                encoded_string = ""                
                for turn in dialog["turns"]:
                    if int(turn["turn_id"]) % 2 == 0:
                        utt = normalize(turn["utterance"])
                        encoded_string += "<sos_u> " + utt + " <eos_u> <sos_b> "
                        state = parse_state(turn)
                        for key, value in state.items():
                            special_tokens.add(f"[{key}]")
                            encoded_string += f"[{key}] "
                            encoded_string += " ".join(value["requested_slots"]) + " "
                            for k2,v2 in value.items():
                                if k2 != "requested_slots":
                                    encoded_string += "#"+ translate.get(k2, k2) + " " + v2 + " "
                        encoded_string += "<eos_b> "
                    else:
                        dialog_acts = acts[dialog["dialogue_id"]][turn["turn_id"]]
                        act_string = []
                        response = turn["utterance"]
                        for key in dialog_acts["dialog_act"]:
                            act_string.append("["+key.lower()+"]")
                            for param in dialog_acts["dialog_act"][key]:
                                if param[0] != "none":
                                    param[0] = param[0]
                                    act_string.append(" ".join(param))
                                    if param[1] != "?":
                                        special_tokens.add(f"[{param[0]}]")
                                        response = response.replace(param[1], f"[{param[0]}]")
                        encoded_string += "<sos_a> " + " ".join(act_string) + " <eos_a> "
                        encoded_string += "<sos_r>" + response.lower() + "<eos_r>"
                fout.write(json.dumps({"id": dialog["dialogue_id"], "domain": dialog["services"],
                "text": encoded_string})+"\n")
    with open(path/"tokens.txt", "w+") as fout:
        for token in special_tokens:
            fout.write(token)
            fout.write("\n")

if __name__ == "__main__":
    generate_encoded()