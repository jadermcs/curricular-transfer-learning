import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from utils.nlp import parse_state
from utils.db_ops import MultiWozDB
from mwzeval.metrics import Evaluator

np.random.seed(42)
sizencode = 512
device = "cuda"
device = "cpu"
dbs = {
    'attraction': 'data/db/attraction_db_processed.json',
    'hospital': 'data/db/hospital_db_processed.json',
    'hotel': 'data/db/hotel_db_processed.json',
    'police': 'data/db/police_db_processed.json',
    'restaurant': 'data/db/restaurant_db_processed.json',
    'taxi': 'data/db/taxi_db_processed.json',
    'train': 'data/db/train_db_processed.json',
}
mwozdb = MultiWozDB("data/db")


def model_predict(model, tokenizer, device, datasets):
    predicted = {}
    for dialog in tqdm(datasets["test"]):
        did = dialog["id"].lower().rstrip(".json")
        utterances = dialog["text"].split("<sos_b>")
        predicted[did] = []
        responses = []
        for i in range(len(utterances)-1):
            example = "<sos_b>".join(utterances[:i+1])[-sizencode:]
            responses.append(example)
        encode = tokenizer(responses, return_tensors="pt", truncation=True,
                           padding=True, max_length=sizencode)
        with torch.no_grad():
            encode = {k: v.to(device) for k, v in encode.items()}
            generate = model.generate(
                **encode,
                max_new_tokens=80,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.encode("<eos_b>")[0]
            )
        for gen in generate:
            state = {}
            gen = tokenizer.decode(gen)
            gen = gen.split("<sos_b>")[-1].split("<eos_b>")[0].strip()
            parsed = parse_state(gen)
            print(parsed)
            continue
            response = gen.split("<sos_r>")[-1].split("<eos_r>")[0].strip()
            parsed = gen.split("<sos_b>")[-1].split("<eos_b>")[0].strip()
            parsed = parse_state(parsed)
            for k, v in parsed:
                try:
                    state[k] = v
                except Exception as e:
                    print(k, e)
                    exit()
            predicted[did].append({
                "response": response,
                "state": state,
            })
        exit()
    return predicted


def get_responses_list(predicted_states):
    resp = []
    for dialog in predicted_states.values():
        for turn in dialog:
            resp.append(turn["response"])
    return resp


def main():
    datasets = load_dataset("json", data_files={
            "train": "data/multiwoz/train/encoded.json",
            "valid": "data/multiwoz/dev/encoded.json",
            "test": "data/multiwoz/test/encoded.json",
        })

    models = [
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
    ]
    curriculums = ["", "ta_noencode/", "ta_encode_nolabel/", "ta_encode/"]
    models = [f"models/{name}/{curriculum}multiwoz" for name in models for
              curriculum in curriculums]

    fout = open("metrics1.txt", "w")
    for path in models:
        tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left",
                                                  truncation_side="left")
        model = AutoModelForCausalLM.from_pretrained(path)
        model.to(device)

        predicted = model_predict(model, tokenizer, device, datasets)
        e = Evaluator(bleu=True, success=True, richness=True)
        results = e.evaluate(predicted)

        basic = (results["success"]["inform"]["total"] +
                 results["success"]["success"]["total"])
        combined = results["bleu"]["mwz22"] + .5 * basic
        print(path, file=fout)
        print(results["bleu"]["mwz22"], results["success"]["inform"]["total"],
              results["success"]["success"]["total"], combined, sep=" & ",
              file=fout)


if __name__ == "__main__":
    main()
