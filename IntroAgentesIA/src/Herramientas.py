import json            # Tratamiento de JSON
import re              # Expresiones regulares
import requests        # Para solicitudes http
from datetime import datetime

# Funciones: Herramientas y analizadores
def obtieneAccion(text) -> list[dict]:
    actions = []
    
    # Regex: FUNC(nombre:parametros)
    #matches = re.findall(r'FUNC\((\w+)(?::(.+))?\)', text)
    matches = re.findall(r'FUNC\((\w+)(?::(.+))?\)', text, flags=re.IGNORECASE)
    
    for nombre, param in matches:
        #print(f"Nombre de la funcion: {nombre}")
        #print(f"Los parametros son: {param}")
        actions.append({'accion': nombre, 'param': param})
    
    return actions


############### HERRAMIENTAS #############################

#######
#   Funcion tonta que simula la conexion al clima
#####
def tontaClima(location):
    res = f"{location} es muy fria en Verano"
    return res

#######
#   Funcion para obtener el clima que necesita conexión
#####
def queTiempoHace(location):

  url = f"https://wttr.in/{location}"
  params = {
      "format": "j1"   # salida JSON estructurada
  }
  resp = requests.get(url, params=params, timeout=10)
  resp.raise_for_status()
  data = resp.json()

  # Zona "current_condition" tiene el tiempo actual
  current = data["current_condition"][0]

  temp_c = float(current["temp_C"])         # Temperatura
  sensacionTermica = float(current["FeelsLikeC"])
  humedad = int(current["humidity"])
  precipitación = float(current["precipMM"])

  # 'weatherDesc' es una lista de diccionarios con 'value' (texto)
  descripcion = current["weatherDesc"][0]["value"]  # p.ej. "Partly cloudy"

  # Lluvia esperada hoy (mm) → primer bloque de 'weather'
  today = data["weather"][0]
  rain_mm = float(today["hourly"][0]["precipMM"])
  probabilidad = float(today["hourly"][0]["chanceofrain"])

  return f"{location}: {temp_c}°C, sensación termica {sensacionTermica}, humedad {humedad}%, lluvias {precipitación}, {descripcion}\n"

# PARSER CORREGIDO - raw strings dobles (r''')
def extract_action(text):
    patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',      # JSON simple
        r'``````',             # Bloque markdown
        r'Action:\s*(\{.*?\})',                   # Action: {...}
        r'"action":\s*"([^"]+)"',                 # Solo nombre acción
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1) if match.lastindex else match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Si falla JSON completo, prueba solo el action name
                if '"action"' in json_str:
                    return {"action": re.search(r'"action":\s*"([^"]+)"', json_str).group(1)}
    return None

#######
#   Obtenemos la fecha y la hora
#####
def fechaHora():
    ahora = datetime.now()
    hora = ahora.strftime("%H:%M")
    fecha = ahora.strftime("%d-%m-%Y")
    return f"Fecha y Hora: {fecha} {hora}"

