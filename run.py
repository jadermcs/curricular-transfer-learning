from train import mwoz, ta_encode, ta_noencode
from utils import multiwoz, tripadvisor, tripadvisor_noencode
from eval import main as main_eval

TOKEN_LENGTH = "512"


FRACTION = [
    5, 10, 20, 50
]

GPT_LIST = [
    ("gpt2", "60000", "16", "4", "200"),
    ("gpt2-medium", "120000", "4", "16", "200"),
    #("gpt2-large", "100000", "2", "32", "200"),
    #("gpt2-xl", "500000"),
]

if __name__ == "__main__":
    print("Processing tripadvisor...")
    tripadvisor.main()
    print("Processing tripadvisor (no encode)...")
    tripadvisor_noencode.main()
    print("Processing multiwoz...")
    multiwoz.generate_encoded()
    exit()

    for model_type, max_steps, BATCH_SIZE, GRAD_ACC, EPOCHS in GPT_LIST:
        # gpt-2 -> multiwoz
        mwoz.main([
            "--directory", f"models/{model_type}/multiwoz",
            "--checkpoint", f"{model_type}",
            "--num_train_epochs", EPOCHS,
            "--batch_size", BATCH_SIZE,
            "--gradient_accumulation_steps", GRAD_ACC,
            "--token_length", TOKEN_LENGTH,
        ])

        # gpt-2 -> tripadvisor (with transform)
        # ta_encode.main([
        #         "--directory", f"models/{model_type}/ta_encode",
        #         "--checkpoint", f"{model_type}",
        #         "--batch_size", BATCH_SIZE,
        #         "--gradient_accumulation_steps", GRAD_ACC,
        #         "--token_length", TOKEN_LENGTH,
        #         "--max_steps", max_steps,
        # ])

        # # gpt-2 -> tripadvisor (without tods transformation) -> multiwoz
        # ta_noencode.main([
        #         "--directory", f"models/{model_type}/ta_noencode",
        #         "--checkpoint", f"{model_type}",
        #         "--batch_size", BATCH_SIZE,
        #         "--gradient_accumulation_steps", GRAD_ACC,
        #         "--token_length", TOKEN_LENGTH,
        #         "--max_steps", max_steps,
        # ])

        # gpt-2 -> tripadvisor (both) -> multiwoz
        for encode in ["encode", "noencode"]:
            mwoz.main([
                "--directory", f"models/{model_type}/ta_{encode}/multiwoz",
                "--checkpoint", f"models/{model_type}/ta_{encode}",
                "--num_train_epochs", EPOCHS,
                "--batch_size", BATCH_SIZE,
                "--gradient_accumulation_steps", GRAD_ACC,
                "--token_length", TOKEN_LENGTH,
            ])

        # gpt-2 -> tripadvisor -> multiwoz (low resource setting)
        # for frac in FRACTION:
        #     mwoz.main([
        #         "--directory", f"models/{model_type}/ta_encode/sgd/multiwoz_{frac}",
        #         "--checkpoint", f"models/{model_type}/ta_encode/sgd",
        #         "--num_train_epochs", EPOCHS,
        #         "--batch_size", BATCH_SIZE,
        #         "--gradient_accumulation_steps", GRAD_ACC,
        #         "--token_length", TOKEN_LENGTH,
        #         "--percent", frac,
        #     ])
    #run evaluation
    main_eval()


# TODO:

# sgd script:
# preprocess data

# tripadvisor:
# add special tokens from NER