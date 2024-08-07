import base64
import json
import os

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Tokenizer,
)

# Note: code here is very unrefined and sloppy; primarily used for
# getting POC. Tons of things for the linter to complain about here,
# but is that a big deal for a test script?

model_dir = "../downloaded_models/model"
serialized_dir = "../downloaded_models/tensorized"


def load_model(model_dir, split=None, cache_8bit=True, serialize=False,
               use_tensorizer=False):
    global model, config, tokenizer, cache

    config = ExLlamaV2Config()
    config.model_dir = model_dir
    config.write_state_dict = serialize
    config.load_with_tensorizer = use_tensorizer
    if config.load_with_tensorizer:
        config.serialized_dir = serialized_dir

    config.prepare()

    model = ExLlamaV2(config)
    print(" -- Loading model: " + model_dir)

    model.load(split)

    tokenizer = ExLlamaV2Tokenizer(config)

    if serialize:
        from util.tensorizer_utils import serialize
        serialize(model, serialized_dir)

    cache = ExLlamaV2Cache_8bit(model, batch_size=4)
    return model, tokenizer, cache


def get_tensors(module):
    k = module.key
    if hasattr(module, "get_weight"):
        w = module.get_weight()
        if isinstance(w, tuple):
            yield (f"{k}.weight", w[0])
            yield (f"{k}.bias", w[1])
        else:
            yield (f"{k}.weight", w)
    for submodule in module.submodules:
        yield from get_tensors(submodule)


def extract_tensors(model):
    for module in model.modules:
        yield from get_tensors(module)


import hashlib


def tensor_hash(t):
    mv = t.cpu().numpy().data
    h = hashlib.sha256()
    h.update(str(t.dtype).encode("ascii"))
    h.update(b"\0")
    h.update(str(tuple(t.size())).encode("ascii"))
    h.update(b"\0")
    h.update(mv)
    return h.digest()


# The next two tests are to save tensors for comparison
# in `test_tensorizer_loaded_is_same_model`, and are meant to be
# ran, rather ungracefully, in separate processes,
# for memory management reasons
def test_get_hashed_tensors_from_normal_model():
    model, tokenizer, cache = load_model(model_dir=model_dir, serialize=True)

    marker = "normal"

    state_dict = dict(extract_tensors(model))

    hash_dict = {k: base64.b64encode(tensor_hash(v)) for k, v in
                 state_dict.items()}

    with open(f'{model_dir}/hash_dict_{marker}.json', "w") as file:
        json.dump({k: v.decode("ascii") for k, v in hash_dict.items()}, file)

    ## Lazy assertion here but mostly just to save the tensors
    assert hash_dict


def test_get_hashed_tensors_from_deserialized_model():
    model, tokenizer, cache = load_model(model_dir=serialized_dir,
                                         use_tensorizer=True)

    marker = "tensorized"

    state_dict = dict(extract_tensors(model))

    hash_dict = {k: base64.b64encode(tensor_hash(v)) for k, v in
                 state_dict.items()}

    with open(f'{serialized_dir}/hash_dict_{marker}.json', "w") as file:
        json.dump({k: v.decode("ascii") for k, v in hash_dict.items()}, file)

    ## Lazy assertion here but mostly just to save the tensors
    assert hash_dict


def test_tensorizer_loaded_is_same_model():
    hash_dict_a = json.load(open(f'{model_dir}/hash_dict_normal.json'))
    hash_dict_b = json.load(open(f'{serialized_dir}/hash_dict_tensorized.json'))

    assert hash_dict_a == hash_dict_b

# TODO: Add test for asserting deserialized config sameness with original config
def test_serializing_s3():
    s3_path = os.environ["S3_URI"] if "S3_URI" in os.environ else None
    s3_access_key_id = os.environ["S3_ACCESS_KEY_ID"] if "S3_ACCESS_KEY_ID" in os.environ else None
    s3_secret_access_key = os.environ["S3_SECRET_ACCESS_KEY"] if "S3_SECRET_ACCESS_KEY" in os.environ else None
    s3_endpoint = os.environ["S3_ENDPOINT_URL"] if "S3_ENDPOINT_URL" in os.environ else None

    from util.tensorizer_utils import serialize

    config = ExLlamaV2Config()
    config.model_dir = model_dir
    config.write_state_dict = True

    config.prepare()

    model = ExLlamaV2(config)
    model.load()

    serialize(model,
              s3_path,
              s3_access_key_id=s3_access_key_id,
              s3_secret_access_key=s3_secret_access_key,
              s3_endpoint=s3_endpoint
              )

def test_deserialize_s3():
    s3_path = os.environ["S3_URI"] if "S3_URI" in os.environ else None
    s3_access_key_id = os.environ["S3_ACCESS_KEY_ID"] if "S3_ACCESS_KEY_ID" in os.environ else None
    s3_secret_access_key = os.environ["S3_SECRET_ACCESS_KEY"] if "S3_SECRET_ACCESS_KEY" in os.environ else None
    s3_endpoint = os.environ["S3_ENDPOINT_URL"] if "S3_ENDPOINT_URL" in os.environ else None

    from util.tensorizer_utils import deserialize_with_tensorizer

    model = deserialize_with_tensorizer(s3_path,
                                         s3_access_key_id=s3_access_key_id,
                                         s3_secret_access_key=s3_secret_access_key,
                                         s3_endpoint=s3_endpoint
                                         )

    assert model