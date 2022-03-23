from src import mwoz_subset, mwoz, ta_encode, ta_noencode

DOMAINS = [
    "hotel", "train", "attraction", "restaurant",
    "hospital", "taxi", "bus", "police"
]

GPT_LIST = [
    "gpt2-small",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
]

if __name__ == "__main__":
    for model_type in GPT_LIST:
        # gpt-2 -> multiwoz(subset)
        for i in DOMAINS:
            mwoz_subset.main([
                "--directory", f"models/mwoz_subset_{i}",
                "--checkpoint", f"{model_type}",
            ])

        # gpt-2 -> multiwoz -> multiwoz(subset)
        for i in DOMAINS:
            mwoz.main([
                "--directory", f"models/mwoz_{i}",
                "--checkpoint", f"{model_type}"
            ])
            mwoz_subset.main([
                "--directory", f"models/mwoz_{i}/mwoz_subset_{i}",
                "--checkpoint", f"models/mwoz_{i}"
            ])
        # gpt-2 -> tripadvisor -> multiwoz -> multiwoz(subset)
        ta_encode.main([
                "--directory", "models/ta_encode",
                "--checkpoint", f"{model_type}"
        ])

        for i in DOMAINS:
            mwoz.main([
                "--directory", f"models/ta_encode/mwoz_{i}",
                "--checkpoint", "models/ta_encode"
            ])
            mwoz_subset.main([
                "--directory", f"models/ta_encode/mwoz_{i}/mwoz_subset_{i}",
                "--checkpoint", f"models/ta_encode/mwoz_{i}"
            ])
        
        # gpt-2 -> tripadvisor (without tods transformation) -> multiwoz (complete)
        ta_noencode.main([
                "--directory", f"models/ta_noencode",
                "--checkpoint", f"{model_type}"
        ])

        for encode in ["endode", "noencode"]:
            mwoz.main([
                "--directory", f"models/ta_{encode}/mwoz_complete",
                "--checkpoint", f"models/ta_{encode}"
            ])


# mwoz-subset script:
# load previous model;
# select a single domain train eval;
# save learning curve.

# mwoz script:
# load previous model;
# select the domains that will not be selected by the subset train eval;
# save learning curve.

# tripadvisor:
# save learning curve.

