import os
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import torch

def cargaModelo(cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(
        cache_dir, 
        local_files_only =True)

    modelo = AutoModelForCausalLM.from_pretrained(
           cache_dir,  
           dtype=torch.float16,
           device_map="auto",
           local_files_only=True)

    return tokenizer, modelo


######################## Carga de modelo con pipeline es menos flexible
def cargaModeloPiPe(cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(
        cache_dir, 
        local_files_only =True)

    generator = pipeline(
        "text-generation",
        model=cache_dir,
        tokenizer=cache_dir,  # Usa el mismo dir
        dtype=torch.float16,
        device_map="auto",
        #local_files_only=True
    )

    return tokenizer, generator
