# Explore Models
This repo is meant ot provibe some basic (and hopefully) sophisticated 
routines for anaylzing `torch.nn.Modules` and `transformers.PreTrained`
models.


## Dependencies: 
See `requirements.txt`. However, most of it it dervied from `torch` and `transformers` are the 

## Hanlding HuggingFace Cache: 
Note, when pulling model from HuggingFace will result in growing cache at `~/.cache/huggingface`. 

It is advised to manage this cache. Please refer to the [HuggingFace docs](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache). A useful command 
to get familiar with is, `huggingface-cli --delete-cache`.  

---

# General usage: 
The expected way to use it is in interactive mode: 
```
$ python3 -i explore_model.py
```
The tool will prompt with with all the models found in 
`from models import MODELS` (aka, the keys). A futre enhancment
is to allow direct passage of `--model` and `-tokenizer`. 

The `models` file is intended for users to populate with models 
they commonly want to work with. 

After a model is pik, the tool with load it and put you into 
the standard Python Interacive mode. It will keep model in 
memory for you, accessible via `Model` variable. Likewise, 
`Tokenizer` will also be available to the user. 

HuggingFace tokens can be passed in via CLI: 
```
$ python3 -i explore_models --token hf_***
```

If while you are in the interactive mode, you want to restart, you 
can simply call the function `rerun()`. This will re-invoke the 
model selection prompt - without closing the program. A new token 
can be passed via `rerun(token)`. 
