import base64
import json
import os
import time

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

# Note: code here is very unrefined and sloppy; primarily used for
# getting POC. Tons of things for the linter to complain about here,
# but is that a big deal for a test script?

model_dir = "../downloaded_models/model"
serialized_dir = "../downloaded_models/tensorized"
import torch
import gc


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
        from util.serialize_with_tensorizer import serialize
        serialize(model, serialized_dir)

    cache = ExLlamaV2Cache_8bit(model, batch_size=4)
    return model, tokenizer, cache


def tensorizer_load():
    os.environ['TENSORIZER'] = '1'
    os.environ[
        'TENSORIZER_LOC'] = "../downloaded_models/tensorized/serialized_llama_state_dict.tensors"
    gc.collect()
    torch.cuda.empty_cache()

    model_dir = "../downloaded_models/tensorized/"
    model_a = load_model(model_dir=model_dir)


def gen(model, tokenizer, cache, prompt, max_new_tokens):
    print("--------------------------------")
    print("Generating, normal")
    print()

    generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0  # Keep it purely deterministic to ensure weights are the same
    settings.top_k = 50
    settings.top_p = 0.8
    settings.top_a = 0.0
    settings.token_repetition_penalty = 1.15
    settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

    generator.warmup()
    time_begin = time.time()

    output = generator.generate_simple(prompt, settings, max_new_tokens,
                                       seed=1234)

    time_end = time.time()
    time_total = time_end - time_begin

    print(output)
    return output


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
