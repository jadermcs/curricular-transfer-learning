import torch
import json
from pprint import pprint
from tqdm import tqdm
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
from utils.nlp import parse_state
from mwzeval.metrics import Evaluator
from mauve import compute_mauve

np.random.seed(42)
sizencode = 512
device = "cuda"

def main():
    datasets = load_dataset("json", data_files={
            "train": "data/multiwoz/train/encoded.json",
            "valid": "data/multiwoz/dev/encoded.json",
            "test": "data/multiwoz/test/encoded.json",
        })

    predicted = {}

    for d in datasets["test"]:
        id = d["id"].rstrip(".json").lower()
        turns = []
        for belief in d["text"].split("<sos_b>")[1:]:
            bs = parse_state(belief.split("<eos_b>")[0])
            response = belief.split("<sos_r>")[1].split("<eos_r>")[0]
            state = {"response": response, "state":{}}
            for k,v in bs:
                state["state"][k] = v
            turns.append(state)
        predicted[id] = turns

    def get_responses_list(predicted_states):
        resp = []
        for dialog in predicted_states.values():
            for turn in dialog:
                resp.append(turn["response"])
        return resp

    e = Evaluator(bleu=True, success=True, richness=True)
    results = e.evaluate(predicted)
    print(results)

    original_resp = np.random.choice(get_responses_list(predicted), 5000)

    def model_predict(model, device):
        predicted = {}
        for batch in tqdm(datasets["test"]):
            did = batch["id"].lower().rstrip(".json")
            utterances = batch["text"].split("<sos_r>")
            predicted[did] = []
            responses = []
            for i in range(len(utterances)-1):
                example = "<sos_r>".join(utterances[:i+1])[-sizencode:]
                responses.append(example)
            encode = tokenizer(responses, return_tensors="pt", truncation=True,
                                padding=True, max_length=sizencode)
            with torch.no_grad():
                encode = {k:v.to(device) for k,v in encode.items()}
                generate = model.generate(
                    **encode,
                    max_new_tokens=80,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.encode("<eos_r>")[0]
                )
            for gen in generate:
                state = {}
                gen = tokenizer.decode(gen)
                response = gen.split("<sos_r>")[-1].split("<eos_r>")[0].strip()
                for k,v in parse_state(gen.split("<sos_b>")[-1].split("<eos_b>")[0].strip()):
                    try:
                        state[k] = v
                    except:
                        print(k)
                        exit()
                predicted[did].append({
                    "response": response,
                    "state": state,
                })
        return predicted

    models = [
        "models/distilgpt2/multiwoz",
        "models/distilgpt2/ta_noencode/multiwoz",
        "models/distilgpt2/ta_encode_nolabel/multiwoz",
        "models/distilgpt2/ta_encode/multiwoz",
        "models/gpt2/multiwoz",
        "models/gpt2/ta_noencode/multiwoz",
        "models/gpt2/ta_encode/multiwoz",
        "models/gpt2-medium/multiwoz",
        "models/gpt2-medium/ta_noencode/multiwoz",
        "models/gpt2-medium/ta_encode/multiwoz",
        "models/gpt2-large/multiwoz",
        "models/gpt2-large/ta_noencode/multiwoz",
        "models/gpt2-large/ta_encode/multiwoz",
    ]
    # tmp = []
    # for name in models:
    #     for percent in [5,10,20,50]:
    #         tmp.append(f"{name}_{percent}")
    # models = tmp

    fout = open("metrics1.txt", "w")
    for path in models:
        tokenizer = GPT2Tokenizer.from_pretrained(path, padding_side="left", truncation_side="left")
        model = GPT2LMHeadModel.from_pretrained(path)
        model.to(device)

        predicted = model_predict(model, device)
        e = Evaluator(bleu=True, success=True, richness=True)
        results = e.evaluate(predicted)

        #pred_resp = np.random.choice(get_responses_list(predicted), 5000)
        #pred_tokenized = tokenizer(pred_resp.tolist(),return_tensors="pt", truncation=True,
        #                            padding=True, max_length=sizencode).input_ids
        #original_tokenized = tokenizer(original_resp.tolist(),return_tensors="pt", truncation=True,
        #                                padding=True, max_length=sizencode).input_ids
        #out = compute_mauve(
        #    p_tokens=original_tokenized,
        #    q_tokens=pred_tokenized,
        #    device_id=0, num_buckets=500, max_text_length=sizencode, mauve_scaling_factor=1)
        combined = results["bleu"]["mwz22"]+.5*(results["success"]["inform"]["total"]+results["success"]["success"]["total"])
        print(path, file=fout)
        print(results["bleu"]["mwz22"], results["success"]["inform"]["total"], results["success"]["success"]["total"], combined,sep=" & ", file=fout)
        #print(out.mauve, results["richness"]["num_unigrams"], results["richness"]["entropy"], sep=" & ", file=fout)

if __name__ == "__main__":
    main()
