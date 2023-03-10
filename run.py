from train import mwoz, ta_encode, ta_noencode
from utils import multiwoz, tripadvisor, tripadvisor_noencode
from eval import main as main_eval

TOKEN_LENGTH = "512"


FRACTION = [
    "5", "10", "20", "50"
]

GPT_LIST = [
    # ("distilgpt2", "120000", "16", "4", "200"),
    # ("gpt2", "60000", "16", "4", "200"),
    # ("gpt2-medium", "120000", "4", "16", "200"),
    # ("gpt2-large", "100000", "2", "32", "200"),
    ("EleutherAI/pythia-1b-deduped", "100000", "8", "4", "200"),
]

if __name__ == "__main__":
    # print("Processing tripadvisor...")
    # tripadvisor.main()
    # print("Processing tripadvisor (no label)...")
    # tripadvisor.main(label=False)
    # print("Processing tripadvisor (no encode)...")
    # tripadvisor_noencode.main()
    # print("Processing multiwoz...")
    # multiwoz.generate_encoded()

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

        # gpt-2 -> tripadvisor (with transform and label)
        ta_encode.main([
                "--pseudo-intent",
                "--directory", f"models/{model_type}/ta_encode",
                "--checkpoint", f"{model_type}",
                "--batch_size", BATCH_SIZE,
                "--gradient_accumulation_steps", GRAD_ACC,
                "--token_length", TOKEN_LENGTH,
                "--max_steps", max_steps,
        ])

        # gpt-2 -> tripadvisor (with transform no label)
        ta_encode.main([
                "--directory", f"models/{model_type}/ta_encode_nolabel",
                "--checkpoint", f"{model_type}",
                "--batch_size", BATCH_SIZE,
                "--gradient_accumulation_steps", GRAD_ACC,
                "--token_length", TOKEN_LENGTH,
                "--max_steps", max_steps,
        ])

        # gpt-2 -> tripadvisor (without tods transformation) -> multiwoz
        ta_noencode.main([
                "--directory", f"models/{model_type}/ta_noencode",
                "--checkpoint", f"{model_type}",
                "--batch_size", BATCH_SIZE,
                "--gradient_accumulation_steps", GRAD_ACC,
                "--token_length", TOKEN_LENGTH,
                "--max_steps", max_steps,
        ])

        # gpt-2 -> tripadvisor (both) -> multiwoz
        for encode in ["encode", "noencode", "encode_nolabel"]:
            mwoz.main([
                "--directory", f"models/{model_type}/ta_{encode}/multiwoz",
                "--checkpoint", f"models/{model_type}/ta_{encode}",
                "--num_train_epochs", EPOCHS,
                "--batch_size", BATCH_SIZE,
                "--gradient_accumulation_steps", GRAD_ACC,
                "--token_length", TOKEN_LENGTH,
            ])

        # gpt-2 -> multiwoz (low resource setting)
        # for frac in FRACTION:
        #     mwoz.main([
        #         "--directory", f"models/{model_type}/multiwoz_{frac}",
        #         "--checkpoint", f"{model_type}",
        #         "--num_train_epochs", EPOCHS,
        #         "--batch_size", BATCH_SIZE,
        #         "--gradient_accumulation_steps", GRAD_ACC,
        #         "--token_length", TOKEN_LENGTH,
        #         "--percent", frac,
        #     ])

        # gpt-2 -> tripadvisor -> multiwoz (low resource setting)
        # for encode in ["encode", "noencode"]:
        #     for frac in FRACTION:
        #         mwoz.main([
        #             "--directory", f"models/{model_type}/ta_{encode}/multiwoz_{frac}",
        #             "--checkpoint", f"models/{model_type}/ta_{encode}/",
        #             "--num_train_epochs", EPOCHS,
        #             "--batch_size", BATCH_SIZE,
        #             "--gradient_accumulation_steps", GRAD_ACC,
        #             "--token_length", TOKEN_LENGTH,
        #             "--percent", frac,
        #         ])
    # run evaluation
    main_eval()
