import argparse

from xdiffusion.training.image.train import train


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_training_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--sample_with_guidance", action="store_true")
    parser.add_argument("--save_and_sample_every_n", type=int, default=1000)
    parser.add_argument("--load_model_weights_from_checkpoint", type=str, default="")
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--mixed_precision", type=str, default="")
    parser.add_argument("--use_lora_training", action="store_true")

    args = parser.parse_args()

    train(
        num_training_steps=args.num_training_steps,
        batch_size=args.batch_size,
        config_path=args.config_path,
        sample_with_guidance=args.sample_with_guidance,
        save_and_sample_every_n=args.save_and_sample_every_n,
        load_model_weights_from_checkpoint=args.load_model_weights_from_checkpoint,
        resume_from=args.resume_from,
        output_path="output",
        dataset_name=args.dataset_name,
        mixed_precision=args.mixed_precision,
        use_lora_training=args.use_lora_training,
    )


if __name__ == "__main__":
    main()
