# Explore Models
This repo is meant ot provibe some basic (and hopefully) sophisticated 
routines for anaylzing `torch.nn.Modules` and `transformers.PreTrained`
models.


## Dependencies: 
See `requirements.txt`. However, most of it it dervied from `torch` and `transformers` are the 

## Hanlding HuggingFace Cache: 
Note, when pulling model from HuggingFace will result in growing cache at `~/.cache/huggingface`. 

It is advised to manage this cache. Please refer to the [HuggingFace docs]. A useful command 
to get familiar with is, `huggingface-cli --delete-cache`.  
