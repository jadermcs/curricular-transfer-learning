from train import mwoz, sgd, ta_encode, ta_noencode
from process import multiwoz, tripadvisor, tripadvisor_noencode

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
    tripadvisor.main()
    tripadvisor_noencode.main()
    multiwoz.main()

    for model_type in GPT_LIST:
        # gpt-2 -> multiwoz
        # mwoz.main([
        #     "--directory", f"models/{model_type}/multiwoz",
        #     "--checkpoint", f"{model_type}",
        # ])

        # gpt-2 -> sgd -> multiwoz
        # sgd.main([
        #     "--directory", f"models/{model_type}/sgd",
        #     "--checkpoint", f"{model_type}",
        # ])
        # mwoz.main([
        #     "--directory", f"models/{model_type}/sgd/multiwoz",
        #     "--checkpoint", f"models/{model_type}/sgd",
        # ])
        
        # gpt-2 -> tripadvisor -> sgd -> multiwoz
        ta_encode.main([
                "--directory", f"models/{model_type}/ta_encode",
                "--checkpoint", f"{model_type}",
        ])
        # sgd.main([
        #     "--directory", f"models/{model_type}/ta_encode/sgd",
        #     "--checkpoint", f"models/{model_type}/ta_encode",
        # ])
        # mwoz.main([
        #     "--directory", f"models/{model_type}/ta_encode/sgd/multiwoz",
        #     "--checkpoint", f"models/{model_type}/ta_encode/sgd",
        # ])
        
        # gpt-2 -> tripadvisor (without tods transformation) -> multiwoz
        ta_noencode.main([
                "--directory", f"models/{model_type}/ta_noencode",
                "--checkpoint", f"{model_type}",
        ])

        for encode in ["endode", "noencode"]:
            mwoz.main([
                "--directory", f"models/{model_type}/ta_{encode}/multiwoz",
                "--checkpoint", f"models/{model_type}/ta_{encode}",
            ])

        # gpt-2 -> tripadvisor -> sgd -> multiwoz (low resource setting)
        # for frac in FRACTION:
        #     mwoz.main([
        #         "--directory", f"models/{model_type}/ta_encode/sgd/multiwoz_{frac}",
        #         "--checkpoint", f"models/{model_type}/ta_encode/sgd",
        #         "--percent", frac,
        #     ])


# TODO:
# mwoz script:
# add mapping.par and delexicalize.py
# save learning curve.

# sgd script:
# save learning curve.

# tripadvisor:
# navigate more pages (in replies) and more cities; london, etc
# add special tokens from NER
# save learning curve.

