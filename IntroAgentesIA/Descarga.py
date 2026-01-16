#!./ia/bin/python
# -*- coding: utf-8 -*-
################################################################################
#  Fco. Javier Rodriguez Navarro 
#  https://www.pinguytaz.net
#
#  Descarga.py: Se realiza una descarga de un modelo, en .env
#     especificamos el modelo y donde descargarlo
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
from huggingface_hub import snapshot_download  # Descarga TODO

def main(argv):
    load_dotenv()
    MODELO=os.getenv('MODELO')
    HF_TOKEN=os.getenv('HF_TOKEN')
    cache_dir=os.getenv('CACHE_DIR')
    os.environ["HF_HOME"] = cache_dir

    snapshot_download(repo_id=MODELO,
           local_dir=cache_dir,  # Copia completa local
           #local_dir_use_symlinks=False  # Archivos reales
    )
    print("Descargado completo a:", cache_dir)

################# Lanzamiento de la funcion principal ##########################
if __name__ == "__main__":
    main(sys.argv)
