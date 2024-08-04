import os

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer, \
    ExLlamaV2Cache_8bit
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler


def serialize(model, serialized_dir: str = None):
    from tensorizer import TensorSerializer
    import shutil

    if not model.config.write_state_dict:
        raise ValueError("Model was not loaded with write_state_dict=True, "
                         "which is necessary for serialization. ")

    if not serialized_dir:
        serialized_dir = model.config.serialized_dir
    os.path.join(model.config.model_dir, "config.json")
    serializer = TensorSerializer(
        os.path.join(serialized_dir, "model.tensors"))
    serializer.write_state_dict(model.state_dict)
    serializer.close()

    # TODO: Should other artifacts be copied? `config.json` is all
    #       that is needed for model loading, but other files are needed
    #       for forward passes like the tokenizer etc
    if not "config.json" in os.listdir(serialized_dir):
        shutil.copyfile(os.path.join(model.config.model_dir, "config.json"),
                        os.path.join(serialized_dir, "config.json"))


## Deserialization example

def deserialize_with_tensorizer(model_dir: str):
    config = ExLlamaV2Config()
    config.model_dir = model_dir
    config.load_with_tensorizer = True
    config.prepare()

    model = ExLlamaV2(config)
    model.load()

    tokenizer = ExLlamaV2Tokenizer(config)

    cache = ExLlamaV2Cache_8bit(model, batch_size=4)

    generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

    generator.warmup()

    print(f" -- Generating...")
    print()

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 1.0
    settings.top_k = 0
    settings.top_p = 0.8
    settings.token_repetition_penalty = 1.02
    settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])


    output = generator.generate_simple("Once upon a time,", settings, 100,
                                       token_healing=True)
    print(output)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Serialize a model")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="The directory containing the model artifacts")
    parser.add_argument("--serialized-dir", type=str, required=False,
                        help="The directory to serialize the model to. "
                             "Defaults to model_dir")
    parser.add_argument("--deserialize", action="store_true",
                        help="Test deserialized model outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = args.model_dir
    serialized_dir = args.serialized_dir
    if not serialized_dir:
        serialized_dir = model_dir

    if args.deserialize:
        deserialize_with_tensorizer(serialized_dir)
        return

    config = ExLlamaV2Config()
    config.model_dir = model_dir
    config.write_state_dict = True
    if config.load_with_tensorizer:
        config.serialized_dir = serialized_dir

    config.prepare()

    model = ExLlamaV2(config)
    model.load()

    serialize(model, serialized_dir)


if __name__ == "__main__":
    main()


