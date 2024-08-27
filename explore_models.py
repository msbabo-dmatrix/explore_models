# ====================================================================|=======:
import os 
import sys 
import argparse
import readline
from collections import OrderedDict
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
from models import MODELS

sol = None 
if os.path.exists("sol_utils.py"): 
    import sol_utils as sol 

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
def load_model(model : str, tokenizer: str, revision = None, token=None ): 
    print("\n\n>>>>>>>>>>  LOADING MODEL >>>>>>>>>>>>>>")
    print("model-name      : %s"%(model))
    print("model-tokenizer : %s"%(tokenizer))
    print("model-revision  : %s"%(revision))
    print("model-token     : %s"%(token))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, 
            revision  = revision, 
            token = token, 
            trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model, 
            token = token,
            revision  = revision, 
            trust_remote_code=True)
    num_params = calc_num_params(model)
    print("\nModel parameters: %s (%s)"%(sizeof_fmt(num_params, suffix=""),
        num_params))
    print("<<<<<<<<<<  MODEL LOADED <<<<<<<<<<<<<<<\n")
    return model, tokenizer, token 
# ====================================================================|=======:
def sizeof_fmt(num, suffix="B"):
    for unit in ("", "K", "M", "G", "T", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        #num /= 1024.0
        num /= 1000.0
    return f"{num:.1f}Yi{suffix}"
# ====================================================================|=======:
# Note, this function is the core method. Therefore, any state setting
# should be managed here (asuming interative_mode execution).
def __interactive_mode(token = None): 
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
                       revision = model_dict.get("revision", None))
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
def __handle_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",action="store_true") 
    parser.add_argument("--token",type=str,default=None)
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
    return args
# ====================================================================|=======:
if __name__ == "__main__": 
    args = __handle_cli_args()
    if sys.flags.interactive: 
        if not args.model: 
            Model, Tokenizer, Token = __interactive_mode(token = args.token)
        elif args.model: 
            Model, Tokenizer, Token = load_model(model=args.model, 
                       tokenizer = args.model,
                       token = args.token, 
                       revision = args.revision)
        else: 
            raise RuntimeError("Something went wrong.")

        print_objs_in_mem()
        print("\n\nNow entering Python Interactive Mode - exit via 'quit()'")
