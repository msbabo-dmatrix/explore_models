# ====================================================================|=======:
import sys 
import argparse
import readline
from collections import OrderedDict
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
# ====================================================================|=======:
MODELS = {
        "Llama-3.1-8B (meta)" : {
            "model": "meta-llama/Meta-Llama-3.1-8B", 
            "tokenizer": "meta-llama/Meta-Llama-3.1-8B", 
        },
        "Gemma-2B  (google)" : {
            "model": "google/gemma-2b", 
            "tokenizer": "google/gemma-2b", 
        },
        "Phi-3 (microsoft)": {
            "model":"microsoft/Phi-3-mini-128k-instruct", 
            "tokenizer": "microsoft/Phi-3-mini-128k-instruct", 
        }, 
        "OPT-125M (facebook)": {
            "model":"facebook/opt-125m", 
            "tokenizer": "facebook/opt-125m", 
        },
        "OPT-125M (d-Matrix)": {
            "model":"d-matrix/opt", 
            "tokenizer": "d-matrix/opt", 
            "revision" : "opt-125m",
        },
        "stablelm-base-alpha-3b (stabilityAI)": {
            "model":"stabilityai/stablelm-base-alpha-3b", 
            "tokenizer": "stabilityai/stablelm-base-alpha-3b", 
        },
        "Sheared-LLaMA-2.7B (Princeton)": {
            "model":"princeton-nlp/Sheared-LLaMA-2.7B", 
            "tokenizer": "princeton-nlp/Sheared-LLaMA-2.7B", 
        }

}
# ====================================================================|=======:
# NOTE: Thisis used in case users can to restart full execution while in 
#       python interactive mode. 
OBJS_IN_MEM = {
    "model" : None, 
    "tokenizer": None
}
# ====================================================================|=======:
def load_model(model : str, tokenizer: str, revision = None, token=None ): 
    print("\n\n>>>>>>>>>>  LOADING MODEL <<<<<<<<<<<<<<")
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
    return model, tokenizer 

# ====================================================================|=======:
def sizeof_fmt(num, suffix="B"):
    for unit in ("", "K", "M", "G", "T", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        #num /= 1024.0
        num /= 1000.0
    return f"{num:.1f}Yi{suffix}"
# ====================================================================|=======:

# ====================================================================|=======:
def _interactive_mode(): 
    print("\n\n=====================================================:")
    print(    "              INTERACTIVE MODEL") 
    print(    "=====================================================:")

    # ----------------------------------------------------------------|-START-:
    # Prompt user for model and load
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

    model, tokenizer = load_model(model=model_dict["model"], 
                       tokenizer = model_dict["tokenizer"],
                       token = model_dict.get("token", None), 
                       revision = model_dict.get("revision", None))

    num_params = calc_num_params(model)
    print(num_params)
    print("\nModel parameters: %s"%sizeof_fmt(num_params, suffix=""))
    # ----------------------------------------------------------------|--END--:

    # ----------------------------------------------------------------|-START-:

    # ----------------------------------------------------------------|--END--:

    # TODO: Create EXIT Baner. Tell users what is available in mem (model, 
    #       tokenizer, etc) and some useful functions at their disposal
    #       create a help_banner and execute before leaving. 
    OBJS_IN_MEM["model"] = model
    OBJS_IN_MEM["tokenizer"] = tokenizer
    return 

# ====================================================================|=======:
def calc_num_params(model : torch.nn.Module): 
    num_params = sum(p.numel() for p in model.parameters())
    return num_params 
# ====================================================================|=======:


# ====================================================================|=======:
def help():
    print("\n\n HELP MENU: ")
    print("(typicaly) objects in memory: ")
# ====================================================================|=======:

# ====================================================================|=START=:
def print_objs_in_mem(): 
    global OBJS_IN_MEM
    for k,v in OBJS_IN_MEM.items(): 
        print(" > %10s : {type=%s}"%(k,type(v)))
# ====================================================================|==END==:
    
# ====================================================================|=START=:
def __handle_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",action="store_true")
    #parser.add_argument("-i", "--interactive",action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__": 
    args = __handle_cli_args()

    if sys.flags.interactive: 
        _interactive_mode()
        print_objs_in_mem()

        print("\nEntering Python Interactive Mode. To exit type 'quit()'")
