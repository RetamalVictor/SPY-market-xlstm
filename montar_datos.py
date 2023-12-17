import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

# hola 

class TimeSeriesDataset(Dataset):
    def __init__(self, train_ratio=0.75, validation_ratio=0.15, limite_capado=3, longitud_secuencia=5):
        # Initialize an empty dataset
        self.X_multivariables = None
        self.label_final = None
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.limite_capado=limite_capado
        self.longitud_secuencia=longitud_secuencia

        self.modo = 'train'

        # Variables para el preprocesado
    def preprocesar_datos(self, datos, nombre_serie):
        """
        Esta funcion transforma una serie de datos de una variable
        Shape [N, ]

        Funcionalidad:
        datos -> hallar incremento:
            (
            incremento,
            incremento_normalizado,
            incremento_noirmalizado_local
            )

        labels -> decalado del sp500 en t+1

        """
        # Recogemos solo los datos numericos y removemos las fechas
        datos_temp = datos.to_numpy()[:, 1]
        # Halla incremento
        _, inc_norm, _ = self.halla_incremento(datos_temp)
        # podemos escoger una o varias de estas 3 como x1
        incremento_acotado = np.clip(-self.limite_capado, self.limite_capado, inc_norm)
        # incremento_acotado = np.maximum(-self.limite_capado, np.minimum(self.limite_capado, inc_norm))
        incremento_cero_uno = incremento_acotado / (self.limite_capado * 2) + 0.5

        label=None
        # var_inc.append(inc)
        if nombre_serie == 'sp50.xlsx':   # inc en fwd será el label
            label = self.shift(incremento_cero_uno, -1)
        # p_mm_norm = halla_posicion(la_serie, n)
        # var_p_mm.append(p_mm_norm)

        # Halla posicion

        return incremento_cero_uno, label

    def halla_incremento(self, serie, n_t=0):
        """
        halla el incremento de la serie: tal cual, normalizado y normalizado local
        la normalización local se hace con la mm del valor absoluto del incremento
        entradas:
            serie, array numpy 1D, la serie a tratar
            n_t, opcional, array numpy 1D, la n para hallar la normalización local
        salidas:
            inc_serie, array numpy 1D, el incremento vulgaris
            inc_serie_norm, array numpy 1D, idem normalizado con media, sigma acumuladas hasta la fecha
            inc_serie_norm_local, array numpy 1D, mismo que antes pero con media acumulada y sigma móvil
        """
        inc_serie = serie / (.00000000000000001 + self.shift(serie, 1)) - 1
        inc_serie[0] = 0
        inc_serie_norm = self.normaliza_movil(inc_serie)
        if n_t == 0:  # no interesa la versión local
            inc_serie_norm_local = inc_serie_norm  # en tal caso, será igual a la otra
        else:
            indice = inc_serie * 0 + 1
            mediaAcum = np.cumsum(inc_serie) / np.cumsum(indice)
            abs = np.abs(inc_serie - mediaAcum)
            sigma_local = abs * 1
            for ii in range(1, len(sigma_local)):
                sigma_local[ii] = (abs[ii] + sigma_local[ii-1]*n_t[ii-1]) / (1+n_t[ii-1])
            inc_serie_norm_local = (inc_serie - mediaAcum) / (.000000000001 + sigma_local)
        return inc_serie, inc_serie_norm, inc_serie_norm_local

    def shift(self, xs, n):
        # desplaza la serie xs en n observaciones; las que faltan se ponen a cero
        e = np.empty_like(xs)
        if n >= 0:
            e[:n] = 0
            e[n:] = xs[:-n]
        else:
            e[n:] = 0
            e[:n] = xs[-n:]
        return e


    def normaliza_movil(self, p_mm):
        # normaliza la entrada (p_mm) restándole la media acumulada de la serie
        # y dividiendo el resultado por la media acumulada del absoluto de las diferencias
        indice = p_mm * 0 + 1
        mediaAcum = np.cumsum(p_mm) / np.cumsum(indice)  # media acumulada variable
        abs = np.abs(p_mm - mediaAcum)
        medAbs = np.cumsum(abs) / np.cumsum(indice)  # media acumulada valor absoluto
        x_t = (p_mm - mediaAcum) / (.000000000000001+medAbs)
        return x_t

    def añadir_datos(self, directorio):

        for esta_serie in os.listdir(directorio):
            # escanea el directorio y lee los excel
            entrada = pd.read_excel(os.path.join(directorio,esta_serie), usecols="A:B")
            # preprocesando los datos, esto debe retornar una serie en torch
            datos, label = self.preprocesar_datos(datos=entrada,nombre_serie=esta_serie)
            datos = datos.astype(np.float32)
            datos_tensor = torch.from_numpy(datos)
            # if label is not None:
            #     label = label.astype(np.float32)
            #     label_tensor = torch.from_numpy(label)
        # extraer datos y label
        # crear secuencias
        # [N, longitud_sequencia, forma_input]
            self.crear_secuencias(datos_tensor, self.longitud_secuencia)

    def crear_secuencias(self, datos, longitud_secuencia):
        X, y = [], [] # Esta lista es interna en la funcion. Solo para una variable
        for i in range(len(datos) - longitud_secuencia + 1):
            seq = datos[i:i + longitud_secuencia]        
            X.append(seq)
            
        X = torch.stack(X) # Convierte la lista de tensores en un tensor
        if self.X_multivariables is None:
            self.X_multivariables = X.unsqueeze(0)
        else:   
            self.X_multivariables = torch.cat((self.X_multivariables, X.unsqueeze(0)))

        # X_bdi =   [
        #       [t-2, t-1, t] -> [t+1]
        #       [t-1, t, t+1] -> [t+2]
        #       [t, t+1, t+2] -> [t+3]
        #       ]
        
        # Tensor_X = [X_sp500; 
        #             X_oil; ...,]
        # Tensor_X = [X_bdi; X_sp500; X_oil; ...,]
        


            
    
    def cambiar_modo(self, modo):
        # Chequea que el modo sea valido

        # Cambia al modo correcto

        pass

    def __len__(self):
    # Devuelde la longitud

        pass

    def __getitem__(self, idx):
    # Primera fase de pruebas:
    ## Devuelve el item en index idx

    # Segunda fase de pruebas
        pass

    def __repr__(self):
        return f'Current Shape {self.X_multivariables.shape}'

dataset = TimeSeriesDataset()
dataset.añadir_datos('dataset')
print(dataset)