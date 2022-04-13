from train import multiwoz, sgd, ta_encode, ta_noencode

FRACTION = [
    5, 10, 20, 50
]

GPT_LIST = [
    "gpt2",
    #"gpt2-medium",
    #"gpt2-large",
    #"gpt2-xl",
]

if __name__ == "__main__":
    for model_type in GPT_LIST:
        # gpt-2 -> multiwoz
        multiwoz.main([
            "--directory", f"models/{model_type}/multiwoz",
            "--checkpoint", f"{model_type}",
        ])

        # gpt-2 -> sgd -> multiwoz
        sgd.main([
            "--directory", f"models/{model_type}/sgd",
            "--checkpoint", f"{model_type}",
        ])
        multiwoz.main([
            "--directory", f"models/{model_type}/sgd/multiwoz",
            "--checkpoint", f"models/{model_type}/sgd",
        ])
        # gpt-2 -> tripadvisor -> sgd -> multiwoz
        ta_encode.main([
                "--directory", f"models/{model_type}/ta_encode",
                "--checkpoint", f"{model_type}",
        ])
        sgd.main([
            "--directory", f"models/{model_type}/ta_encode/sgd",
            "--checkpoint", f"models/{model_type}/ta_encode",
        ])
        multiwoz.main([
            "--directory", f"models/{model_type}/ta_encode/sgd/multiwoz",
            "--checkpoint", f"models/{model_type}/ta_encode/sgd",
        ])
        
        # gpt-2 -> tripadvisor (without tods transformation) -> multiwoz
        ta_noencode.main([
                "--directory", f"models/{model_type}/ta_noencode",
                "--checkpoint", f"{model_type}",
        ])

        for encode in ["endode", "noencode"]:
            sgd.main([
                "--directory", f"models/{model_type}/ta_{encode}/multiwoz",
                "--checkpoint", f"models/{model_type}/ta_{encode}",
            ])

        # gpt-2 -> tripadvisor -> sgd -> multiwoz (low resource setting)
        for frac in FRACTION:
            multiwoz.main([
                "--directory", f"models/{model_type}/ta_encode/sgd/multiwoz_{frac}",
                "--checkpoint", f"models/{model_type}/ta_encode/sgd",
                "--percent", frac,
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

