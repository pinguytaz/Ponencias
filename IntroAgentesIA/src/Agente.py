import os
import sys
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#####################
# agente_react que resuelve la pregunta
#####################
def agente_react(pregunta, tokenizer, model, max_pasos=5):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": pregunta},
    ]

    # Prepara el Prompt en el formato del modelo
    IDsEntradas = tokenizer.apply_chat_template(
        messages,      # Mensaje con roles.
        tokenize=True,  # convertir a ID_TOKENs
        add_generation_prompt=True,   # Añade al final token especiales
        return_tensors="pt",   #Debemos generar un objeto pytorch, necesario par los modelos locales de HuggingFace
        padding=True,     # Rellenar mensajes
        return_dict=True # Pone diccionario no Token
    ).to(model.device)

    #print("IDsEntradas shape:", IDsEntradas.shape)  # torch.Size([1, N])

    # Indicamos que no estamos entrenando el modelo para ahorrar recursos
    #
    
    inicio = time.time()
    with torch.no_grad(): # Ahorro no entrena modelo.
        salidas = model.generate(
            **IDsEntradas,  # Diccionario
            max_new_tokens=50,   #Limite de Tokens nuevos
            temperature=0.25,    # Aleatoriedad 
                                # 0.1 determinista 0.7 medio 1.0 creativo
            top_p = 0.9,         # NUCLEUS escoge grupo de palabras hasta alcanzar 'p' 
            top_k = 3,          # CONTEO corta por num. palabras más probables
            do_sample=True,    # Activa la temperatura False siempre la mas probable
            #pad_token_id=tokenizer.pad_token_id # Evita el warning 
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,  # define TOKEN padding
            eos_token_id=tokenizer.eos_token_id  # Para al encontrar EOS
        )
    fin = time.time()
    tokensGenerados= salidas[0].shape[0] - IDsEntradas["input_ids"].shape[1]
    tiempoTotal = fin - inicio
    tokensPorSegundo = tokensGenerados / tiempoTotal
    print("*** Rendimiento ***")
    print(f"\tVelocidad: {tokensPorSegundo:.2f} tokens/s")
    print(f"\tLatencia total: {tiempoTotal:.2f} segundos")


    # Sin dict es un tensor [IDsEntradas.shape[1]:]  
    # Con dict es un diccionario {"IDsEntradas": Tensor, "attention_mask": Tensor}.
    # salidas[0] primera lista de resultados, primera pregunta
    #salidaAgente = tokenizer.decode(salidas[0][IDsEntradas.shape[1]:], skip_special_tokens=True)    #Respuesta con tokens
    #print(f"****INI_MSG**** {tokenizer.decode(salidas[0])} *****FIN_MSG****")
    salidaAgente = tokenizer.decode(salidas[0][IDsEntradas["input_ids"].shape[1]:], skip_special_tokens=True) # Solo respuesta de dic.

    return salidaAgente, messages



#####################
# agente_reactPiPe resuelve la pregunta mediante carga con pipeline
#####################
def agente_reactPiPe(pregunta, tokenizer, generator):
    messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pregunta},
        ]

    prompt = tokenizer.apply_chat_template(
         messages,
         tokenize=False,
         add_generation_prompt=True)

    salidas = generator(
            #entradas.input_ids,
            prompt, 
            max_new_tokens=50,
            do_sample=True,    
            temperature=0.25, 
            #top_p = 0.9,   
            #top_k = 3,    
            pad_token_id=tokenizer.pad_token_id ,
            return_full_text=False
        )

    salidaAgente = salidas[0]['generated_text'].strip()

    return salidaAgente, messages

############################# PROMPT ##################################################
#################################################################################################  
# Definimos el prompt del rol del sistema indicando herramientas
#	Quien Soy
#	Cual es el objetivo
#	Funciones de las que se dispone 
#       Como nos informa de estas funciones, poniendo ejemplo.
#	Formato de como informarnos de la funcion y parametro a utilizar (Especial, JSON, XML,....
##################################################################################################
SYSTEM_PROMPT = """Eres un asistente que debe responder lo mejor que puedas y puedes llamar a una o mas funciones para ayudar a generar la consulta del usuario. Estas son las funciones disponibles:

FUNC(elTiempo:Madrid)   Nos da el tiempo meteorologico: temperatura, lluvias, viento, nubes, clima
FUNC(laFecha)           Obtenemos la hora y fecha del sistema.

si se pregunta por varia localidades se lanzara una funcion par cada una de ellas.

Ejemplos:
¿Qué tiempo en Madrid? -> FUNC(elTiempo:Madrid)

¿Qué tiempo hace en Madrid y Barcelona? -> FUNC(elTiempo:Madrid) 
FUNC(elTiempo:Barcelona)

¿Qué hora? -> FUNC(laFecha)

¿Tiempo + hora? -> FUNC(elTiempo:Alcobendas) 
FUNC(laFecha)

SIEMPRE usa el siguiente formato:

FUNC(nombre:parametro)

¡Comienza ahora! Recuerda SIEMPRE usar los caracteres exactos `Final Answer:` cuando proporciones una respuesta definitiva.
"""
