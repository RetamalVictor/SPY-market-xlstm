import torch
import torch.nn as nn
import torch.nn.functional as F

class AtencionDeUnaCabeza(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(AtencionDeUnaCabeza, self).__init__()
        self.query_linear = nn.Linear(input_dim, attention_dim)
        self.key_linear = nn.Linear(input_dim, attention_dim)
        self.value_linear = nn.Linear(input_dim, attention_dim)

    def forward(self, query, key, value):
        # Calcular los vectores de consulta, clave y valor
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        # Calcular los puntajes de atención
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (attention_dim ** 0.5)

        # Aplicar softmax para obtener los pesos de atención
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Aplicar los pesos de atención a los valores
        attention_output = torch.matmul(attention_weights, value)

        return attention_output, attention_weights

# Definir dimensiones
input_dim = 10
attention_dim = 5
seq_length = 4
batch_size = 2

# Crear un tensor de entrada aleatorio (batch_size, seq_length, input_dim)
inputs = torch.randn(batch_size, seq_length, input_dim)

# Crear una instancia de la clase AtencionDeUnaCabeza
attention = AtencionDeUnaCabeza(input_dim, attention_dim)

# Aplicar atención
output, weights = attention(inputs, inputs, inputs)

# Imprimir las formas de la salida y los pesos de atención
print("Forma de la salida:", output.shape)  # (batch_size, seq_length, attention_dim)
print("Forma de los pesos de atención:", weights.shape)  # (batch_size, seq_length, seq_length)
