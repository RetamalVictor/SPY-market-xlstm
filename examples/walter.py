import yfinance as yf
import pandas as pd

stock_list = ['SPY', 'QQQ', 'VGK', 'VIXY', 'TLT', 'UUP', 'CPER', 'GLD', 'DBO']
stock_list_plus = stock_list.copy()
stock_list_plus.append('vola')

# todos son ETF's:
# índices de bolsa: spy: s&p500; qqq: nasdaq; vgk: FTSE Europe
# vixy: volatilidad implícita opciones (contratos próximos)
# variables financieras: tlt: 20 yrs treasury bonds; uup: dollar index exchange rate
# materias primas: cper: cobre; gld: oro; pbo: petróleo
data = yf.download(stock_list, start="2012-01-01", end="2024-01-05")

# AQUI PODREMOS CAMBIAR LA VARIABLE LABEL MODIFICANDO label_escogido

def crea_el_incremento(nombre_serie, data):
    # en la misma base de datos crea el incremento de la serie cuyo nombre es nombre_serie
    nuevo_nombre = 'inc_' + nombre_serie
    data[nuevo_nombre] = ( data['Adj Close'][nombre_serie] / data['Adj Close'][nombre_serie].shift() - 1 ).fillna(0)
    
# creamos los incrementos de cada feature 
for nombre in stock_list:
    crea_el_incremento(nombre, data)
    
    # y el label (inc de sp500)