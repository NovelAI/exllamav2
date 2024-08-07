import os
import builtins
from tensorizer import TensorSerializer, stream_io
from functools import partial
from typing import ContextManager

from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer, \
    ExLlamaV2Cache_8bit
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler

def open_wrapper(path, mode, **kwargs):
    if path.startswith("s3://"):
        s3_creds = {
            "s3_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", None),
            "s3_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", None),
            "s3_endpoint": os.environ.get("AWS_ENDPOINT_URL", None),
        }
        return stream_io.open_stream(path, mode, **s3_creds)
    else:
        return open(path, mode)

read_stream, write_stream = (
    partial(
        open_wrapper,
        mode=mode,
    )
    for mode in ("rb", "wb+")
)

def serialize(model, serialized_dir, s3_creds=None):

    local_config_path = os.path.join(model.config.model_dir, "config.json")
    local_tokenizer_json_path = os.path.join(model.config.model_dir, "tokenizer.json")
    local_tokenizer_config_json_path = os.path.join(model.config.model_dir, "tokenizer_config.json")

    for path in [local_config_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    if not os.path.exists(local_tokenizer_config_json_path):
        print(f"tokenizer_config.json not found at {local_tokenizer_config_json_path}. "
              f"Skipping..")

    if not model.config.write_state_dict:
        raise ValueError("Model was not loaded with write_state_dict=True, "
                         "which is necessary for serialization. ")

    if not serialized_dir:
        serialized_dir = model.config.serialized_dir

    s3_creds = {
        "s3_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", None),
        "s3_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", None),
        "s3_endpoint": os.environ.get("AWS_ENDPOINT_URL", None),
    }

    model_uri = os.path.join(serialized_dir, "model.tensors")
    with write_stream(model_uri, **s3_creds) as stream:
        serializer = TensorSerializer(stream)
        serializer.write_state_dict(model.state_dict)
        serializer.close()

    config_path = os.path.join(serialized_dir, "config.json")
    with write_stream(config_path, **s3_creds) as stream:
        with open(local_config_path) as f:
            stream.write(f.read().encode("utf-8"))

    #tokenizer_json_path = os.path.join(serialized_dir, "tokenizer.json")
    #with write_stream(tokenizer_json_path, **creds) as stream:
    #    with open(os.path.join(model.config.model_dir, "tokenizer.json")) as f:
    #        stream.write(f.read().encode("utf-8"))

    #tokenizer_config_json_path = os.path.join(serialized_dir, "tokenizer_config.json")
    #with write_stream(tokenizer_config_json_path, **creds) as stream:
    #    with open(os.path.join(model.config.model_dir, "tokenizer_config.json")) as f:
    #        stream.write(f.read().encode("utf-8"))

    # TODO: Should other artifacts be copied? `config.json` is all
    #       that is needed for model loading, but other files are needed
    #       for forward passes like the tokenizer etc


## Deserialization example

def deserialize_with_tensorizer(model_dir: str, **kwargs):
    config = ExLlamaV2Config()
    config.model_dir = model_dir

    # Also can be enabled by specifying `TENSORIZER` in env vars
    config.load_with_tensorizer = True
    config.prepare()

    for key, value in kwargs.items():
        setattr(config, key, value)

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
    return model

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


def io_handler(use_tensorizer: bool) -> ContextManager:
    return _IOHandlerImpl(use_tensorizer)


class _IOHandlerImpl:

    def __init__(self, use_tensorizer: bool):
        self.use_tensorizer = use_tensorizer
        self._open = builtins.open
        self._file_exists = os.path.exists
        self._path_join = os.path.join

    def modified_open(self, *args, **kwargs):
        tensorizer_args = kwargs.get("tensorizer_args", None)
        if self.use_tensorizer and tensorizer_args:
            assert isinstance(tensorizer_args,
                              dict), "tensorizer_args must be a dict"
            return read_stream(*args, **tensorizer_args)
        else:
            return self._open(*args, **kwargs)

    def modified_file_exists(self, *args, **kwargs):
        if self.use_tensorizer:
            return True
        else:
            return self._file_exists(*args, **kwargs)



    def __enter__(self):
        builtins.open = self.modified_open

    def __exit__(self, exc_type, exc_val, exc_tb):
        builtins.open = self._open



if __name__ == "__main__":
    main()
