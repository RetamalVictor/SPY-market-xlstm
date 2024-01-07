import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, longitud_secuencia):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, output_size)
        self.longitud_secuencia = longitud_secuencia
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=1):
        h0 = torch.zeros(self.longitud_secuencia, batch_size, self.output_size)
        c0 = torch.zeros(self.longitud_secuencia, batch_size, self.output_size)
        return (h0, c0)
    
        # [longitud_seq, num_batch, num_feat]
    def forward(self, input):
        """
        Input tiene forma [long_seq, num_batch, num_feat]
        """
        h, c = self.init_hidden()
        out, (h, c) = self.lstm(input, (h, c))
        return out[:, -1, :] # dia t+1

if __name__ == "__main__":
    input_size = 5
    hidden_size = 1
    output_size = 3
    longitud_secuencia = 3
    modelo_prueba = LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        longitud_secuencia=longitud_secuencia
    )
    print(modelo_prueba)
    
    # punto de dato -> [longitud_seq, 1, num_feat] <- aqui
    # 


# Modelo:
        # input_secuencia -> LSTM
        # -> output(prediccion)


