import numpy as np
import re

def cargar_dataset_descriptores(archivo):
    with open(archivo, "r", encoding="utf-8") as f:
        contenido = f.read()

    # Separar por el delimitador de bloques
    bloques = contenido.strip().split("**************************")
    datos_dict = {}

    for bloque in bloques:
        lineas = bloque.strip().split("\n")
        if len(lineas) < 2:
            continue  # Si no hay suficiente contenido, lo ignoramos

        # Extraer el ID (primera línea)
        try:
            id_elemento = int(lineas[0].strip())
            
        except ValueError:
            print(f"Advertencia: No se pudo convertir el ID en '{lineas[0]}'")
            continue

        # Unimos el contenido numérico
        contenido_numerico = " ".join(lineas[1:])
        contenido_numerico = contenido_numerico.replace("[", " ").replace("]", " ").replace("\n", " ").replace("\t", " ")

        # Reemplazar múltiples espacios por un solo espacio
        contenido_numerico = re.sub(r'\s+', ' ', contenido_numerico).split(" ")

        
        for i in contenido_numerico:
            if len(i) == 0:
                contenido_numerico.remove(i)

        # Convertir la lista de valores en floats
        try:
            valores = list(map(float, contenido_numerico))
        except ValueError as e:
            print(f"Error al convertir valores para el ID {id_elemento}: {e}")
            print(f"Línea problemática: {contenido_numerico[:200]}...")  # Mostrar parte de la línea problemática
            continue

        # Determinar la forma de la matriz según la cantidad de valores
        if len(valores) == 32:
            shape = (2, 2, 2, 4)
        elif len(valores) == 62500:
            shape = (25, 25, 25, 4)
        else:
            print(f"Advertencia: cantidad inesperada de valores ({len(valores)}) para ID {id_elemento}")
            continue

        # Convertir a array numpy con la forma adecuada
        matriz = np.array(valores, dtype=np.float32).reshape(shape)

        # Almacenar en el diccionario
        datos_dict[id_elemento] = matriz

    return datos_dict

    
archivo = "./dataset/dataset_descriptores_float.txt"
datos = cargar_dataset_descriptores(archivo)

# Acceder a una matriz por su ID
#id_ejemplo = 3036
#print(f"Datos para ID {id_ejemplo}:\n", datos[id_ejemplo])


# Iterar sobre todos los valores en la matriz
#for i in datos[id_ejemplo].flatten():  # Convertimos la matriz a una lista 1D
#    if i != 0:
        #print(str(i))