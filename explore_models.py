# ====================================================================|=======:
import os 
import sys 
import re
import inspect
import argparse
import readline
from collections import OrderedDict
import torch 
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from models import MODELS

sol = None 
if os.path.exists("sol_utils.py"): 
    import sol_utils as sol 
# ====================================================================|=======:
def sizeof_fmt(num, suffix="B"):
    for unit in ("", "K", "M", "G", "T", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        #num /= 1024.0
        num /= 1000.0
    return f"{num:.1f}Yi{suffix}"
# ====================================================================|=======:
def get_sig(function): 
    """
    The Signature object represents the call signature of a callable 
    object and its return annotation
    """
    return inspect.signature(function)
# ====================================================================|=======:
def calc_max_kv_cache(model, batch_size = 1, seq_length = None, num_bytes=None): 
    func = "calc_max_kv_cache"
    _16s = ["torch.float16", "torch.bfloat16"]

    print(f"DEBUG [{func}]: type(model) = {type(model)}")

    #if isinstance(model,AutoConfig): 
    #if "PretrainedConfig" in model.__class__.__bases__: 
    if "Config" in str(model.__class__): 
        dHead   = (model.hidden_size / model.num_attention_heads)
        kvHeads = (model.num_key_value_heads)
        layers  = (model.num_hidden_layers)
        seqlen  = (model.max_position_embeddings)
        prec    = (model.torch_dtype)
    else: 
        dHead   = (model.config.hidden_size / model.config.num_attention_heads)
        kvHeads = (model.config.num_key_value_head)
        layers  = (model.config.num_hidden_layers)
        seqlen  = (model.config.max_position_embeddings)

    if seq_length == None: 
        seq_length = seqlen 

    if num_bytes == None: 
        if str(prec) not in _16s: 
            raise RuntimeError("missed dtype extraction")
        num_bytes = 2

    print(f"DEBUG [{func}]: batch_size = {batch_size}")
    print(f"DEBUG [{func}]: seq_length = {seq_length}")
    print(f"DEBUG [{func}]: dHead      = {dHead}")
    print(f"DEBUG [{func}]: kvHeads    = {kvHeads}")
    print(f"DEBUG [{func}]: layers     = {layers}")
    print(f"DEBUG [{func}]: num_bytes  = {num_bytes}")

    result = 2 *(batch_size) * (seq_length) * dHead * kvHeads * layers * num_bytes
    return result

# ====================================================================|=======:
def kv_cache_per_token(model, as_str = True): 
    _16s = ["torch.float16", "torch.bfloat16"]
    if str(model.config.torch_dtype) in _16s: 
        num_bytes = 2
    else: 
        raise RuntimeError("missed dtype extraction")
    hidden_size = model.config.hidden_size
    layers = model.config.num_hidden_layers
    result = 2 * hidden_size * layers * num_bytes
    if as_str: return sizeof_fmt(result)
    return result
# ====================================================================|=======:
def kv_cache_for_max_context_length(model, as_str = True):
    max_cl = model.config.max_position_embeddings
    result = max_cl * kv_cache_per_token(model, False)
    if as_str: return sizeof_fmt(result)
    return result
# ====================================================================|=======:
# MODELS[<new model>] = {"model":..., "tokenizer":...}
# ====================================================================|=======:
Model = None 
Tokenizer = None 
Token = None
# ^^^ These are set via methods. If a user is interactive mode and sets
#     sets these directly, it will mostly likely results in undefined 
#     behavior
# ====================================================================|=======:
# NOTE: Thisis used in case users can to restart full execution while in 
#       python interactive mode. 
OBJS_IN_MEM = {
    "Model" : None, 
    "Tokenizer": None, 
    "Token" : None, # TODO: Need a way to hide the token. 
}
# ====================================================================|=======:
def load_model(model : str, tokenizer: str, revision = None, token=None ,
        no_weights = False): 
    if no_weights: 
        print("\n\n>>>>>>>>>>  LOADING CONFIG >>>>>>>>>>>>>>")
        config = AutoConfig.from_pretrained(model)
        print("<<<<<<<<<<  CONFIG LOADED <<<<<<<<<<<<<<<\n")
        return config, None, token

    print("\n\n>>>>>>>>>>  LOADING MODEL >>>>>>>>>>>>>>")
    print("model-name      : %s"%(model))
    print("model-tokenizer : %s"%(tokenizer))
    print("model-revision  : %s"%(revision))
    if token: 
        print("model-token     : %s"%(token[:5] + "****"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, 
            revision  = revision, 
            token = token, 
            device_map = "auto",
            trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model, 
            token = token,
            device_map = "auto",
            revision  = revision, 
            trust_remote_code=True)
    num_params = calc_num_params(model)
    print("\nModel parameters: %s (%s)"%(sizeof_fmt(num_params, suffix=""),
        num_params))
    print("<<<<<<<<<<  MODEL LOADED <<<<<<<<<<<<<<<\n")
    return model, tokenizer, token 
# ====================================================================|=======:
# ====================================================================|=======:
# Note, this function is the core method. Therefore, any state setting
# should be managed here (asuming interative_mode execution).
def __interactive_mode(token = None, no_weights = False): 
    print("\n\n=====================================================:")
    print(    "                EXPLORE MODELS ") 
    print(    "=====================================================:")
    # ----------------------------------------------------------------|-START-:
    # Prompt user for model and pull from dictionary
    _model_list = list(MODELS.keys())
    print("Please pick a model to load: (select via number)")
    for i, model in enumerate(_model_list, start = 1):
        print("%d.)  %s"%(i, model))
    model_index = int(input("\nmodel index: "))
    model_index = model_index - 1
    if model_index >= len(MODELS): 
        raise ValueError("Model index is out of range: %d"%(model_index + 1))
    model_name = _model_list[model_index]
    model_dict = MODELS[model_name]
    # ----------------------------------------------------------------|--END--:
    if token: model_dict['token'] = token 
    model, tokenizer, token = load_model(model=model_dict["model"], 
                       tokenizer = model_dict["tokenizer"],
                       token = model_dict.get("token", None), 
                       revision = model_dict.get("revision", None),
                       no_weights = no_weights)
    return model, tokenizer , token 
# ====================================================================|=======:
def rerun(token=None): 
    global Model, Tokenizer, Token
    Model, Tokenizer, Token  = __interactive_mode(None) 
# ====================================================================|=======:
def calc_num_params(model : torch.nn.Module): 
    num_params = sum(p.numel() for p in model.parameters())
    return num_params 
# ====================================================================|=======:
def help():
    print("\n\n HELP MENU: ")
    print("(typicaly) objects in memory: ")
# ====================================================================|=======:
def print_objs_in_mem(): 
    global OBJS_IN_MEM, Model, Tokenizer, Token
    OBJS_IN_MEM["Model"] = Model
    OBJS_IN_MEM["Tokenizer"] = Tokenizer
    OBJS_IN_MEM["Token"] = Token
    print("Objects in memory: ")
    for k,v in OBJS_IN_MEM.items(): 
        print(" > %-10s : {type=%s}"%(k,type(v)))
# ====================================================================|=======:
def _extract_token(key): 
    token = None 
    with open(".hf_tokens","r") as fh: 
        for line in fh.readlines(): 
            print(line)
            m = re.search("^%s:\s*(?P<token>[\w]+)"%(key), line)
            if m: 
                token = m.group("token")
                break
    if not token: 
        print(f"Warning - no token found for key: {key}")
    return token
# ====================================================================|=======:
def __handle_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",action="store_true") 
    parser.add_argument("--token",type=str,default=None)
    parser.add_argument("--token-read",type=str,default=None)
    parser.add_argument("--no-weights",action="store_true")

    # TODO: Here is where you would handle token settings.
    # TODO: need to be able to handle users passing in model and tokenizer
    parser.add_argument("--model",type=str,default=None)
    parser.add_argument("--tokenizer",type=str,default=None)
    parser.add_argument("--revision",type=str,default=None)
    args = parser.parse_args()
    # Sanity Checks
    if args.model and not args.tokenizer: 
        print("Warning - no tokenizer provided. Setting tokenizer to model")
        args.tokenizer = args.model

    if args.token_read: 
        args.token = _extract_token(args.token_read)
    return args
# ====================================================================|=======:
if __name__ == "__main__": 
    args = __handle_cli_args()
    if sys.flags.interactive: 
        if not args.model: 
            Model, Tokenizer, Token = __interactive_mode(token = args.token, 
                    no_weights = args.no_weights)
        elif args.model: 
            Model, Tokenizer, Token = load_model(model=args.model, 
                       tokenizer = args.model,
                       token = args.token, 
                       revision = args.revision, 
                    no_weights = args.no_weights)
        else: 
            raise RuntimeError("Something went wrong.")

        print_objs_in_mem()
        print("\n\nNow entering Python Interactive Mode - exit via 'quit()'")
