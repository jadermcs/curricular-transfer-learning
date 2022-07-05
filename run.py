from train import mwoz, sgd, ta_encode, ta_noencode
from process import multiwoz, tripadvisor, tripadvisor_noencode

EPOCHS = 40
BATCH_SIZE = 16


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
    print("Processing tripadvisor...")
    tripadvisor.main()
    print("Processing tripadvisor (no encode)...")
    tripadvisor_noencode.main()
    # print("Processing multiwoz...")
    # multiwoz.main()

    for model_type in GPT_LIST:
        # gpt-2 -> multiwoz
        mwoz.main([
            "--directory", f"models/{model_type}/multiwoz",
            "--checkpoint", f"{model_type}",
            "--num_train_epochs", EPOCHS,
            "--batch_size", BATCH_SIZE,
        ])

        # gpt-2 -> sgd -> multiwoz
        # sgd.main([
        #     "--directory", f"models/{model_type}/sgd",
        #     "--checkpoint", f"{model_type}",
        #     "--num_train_epochs", "40",
        #     "--batch_size", BATCH_SIZE,
        # ])
        # mwoz.main([
        #     "--directory", f"models/{model_type}/sgd/multiwoz",
        #     "--checkpoint", f"models/{model_type}/sgd",
        #     "--num_train_epochs", "40",
        # ])
        
        # gpt-2 -> tripadvisor -> sgd -> multiwoz
        ta_encode.main([
                "--directory", f"models/{model_type}/ta_encode",
                "--checkpoint", f"{model_type}",
                "--num_train_epochs", EPOCHS,
                "--batch_size", BATCH_SIZE,
        ])
        # sgd.main([
        #     "--directory", f"models/{model_type}/ta_encode/sgd",
        #     "--checkpoint", f"models/{model_type}/ta_encode",
        #     "--num_train_epochs", "40",
        # ])
        # mwoz.main([
        #     "--directory", f"models/{model_type}/ta_encode/sgd/multiwoz",
        #     "--checkpoint", f"models/{model_type}/ta_encode/sgd",
        #     "--num_train_epochs", "40",
        # ])
        
        # gpt-2 -> tripadvisor (without tods transformation) -> multiwoz
        ta_noencode.main([
                "--directory", f"models/{model_type}/ta_noencode",
                "--checkpoint", f"{model_type}",
                "--num_train_epochs", EPOCHS,
                "--batch_size", BATCH_SIZE,
        ])

        for encode in ["encode", "noencode"]:
            mwoz.main([
                "--directory", f"models/{model_type}/ta_{encode}/multiwoz",
                "--checkpoint", f"models/{model_type}/ta_{encode}",
                "--num_train_epochs", EPOCHS,
                "--batch_size", BATCH_SIZE,
            ])

        # gpt-2 -> tripadvisor -> sgd -> multiwoz (low resource setting)
        # for frac in FRACTION:
        #     mwoz.main([
        #         "--directory", f"models/{model_type}/ta_encode/sgd/multiwoz_{frac}",
        #         "--checkpoint", f"models/{model_type}/ta_encode/sgd",
        #         "--num_train_epochs", "40",
        #         "--percent", frac,
        #     ])


# TODO:
# mwoz script:
# add mapping.par and delexicalize.py

# sgd script:
# preprocess data

# tripadvisor:
# add special tokens from NER