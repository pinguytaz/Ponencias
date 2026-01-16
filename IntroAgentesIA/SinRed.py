#!./ia/bin/python
# -*- coding: utf-8 -*-
################################################################################
#  Fco. Javier Rodriguez Navarro 
#  https://www.pinguytaz.net
#
#  SinRed.py: Ejemplo de Agente con LLM-QWEN-HuggingFace en local
#       Qwen/Qwen2.5-Coder-1.5B-Instruct
#
#  Historico:
#     - Mes año V1: Enero 2026
#
# Previo:
#    python3 -m venv ia    Genera entorno
#    source ia/bin/activate        Activación del entorno Virtual
#    pip3 install -r requirements.txt     Instalación de dependencias en el entorno activado
################################################################################
# Carga librerias
import os
import sys

from dotenv import load_dotenv
from src.Modelo import cargaModelo
import src.Agente as agente
import src.Herramientas as herramientas

def main(argv):
    load_dotenv()
    #MODELO=os.getenv('MODELO')  
    HF_TOKEN=os.getenv('HF_TOKEN')
    cache_dir=os.getenv('CACHE_DIR')
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = cache_dir

    # Cargamos el Modelo
    tokenizer, modelo = cargaModelo(cache_dir)
    print("Modelo cargado")

    pregunta = input("Pregunta: ")  

    # Ejecutamos el agente, realizando la consulta
    salidaAgente, miMensaje = agente.agente_react(pregunta, tokenizer, modelo) 

    # Ya tenemos la respuesta de que debemos ejecutar
    print("----- Inicia Salidas procesando ----------")
    print("Salida del agente: ",salidaAgente)

    # Extraemos acciones y ejecutamos herramientas
    datosAccion = herramientas.obtieneAccion(salidaAgente)
    res = ""
    if datosAccion:
        print(f"Acciones obtenidas {datosAccion}")
        for accion in datosAccion:
            nomAccion = accion['accion']
            parametro = accion['param']
            match nomAccion:
                case 'laFecha':
                    print(f"\tEjecutamos la accion laFecha()")
                    res = res + herramientas.fechaHora()
                case 'elTiempo':
                    print(f"\tEjecutamos la accion elTiempo({parametro})")
                    res = res + herramientas.tontaClima(parametro)
                    #res = res + herramientas.queTiempoHace(parametro)
                case _:
                    print(f"\tERROR: Funcion no implementada {nomAccion}")
            res = res + "\n"

        print("-----  FIN de las salidas procesadas -----")
        print(f"**** El resultado es: \n{res}")
    else:
        print("+++++++ Sin funciones a ejecutar +++++++")


################# Lanzamiento de la funcion principal ##########################
if __name__ == "__main__":
    main(sys.argv)
