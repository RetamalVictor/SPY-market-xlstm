import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class DirectionalAccuracyLoss(nn.Module):
    def __init__(self):
        super(DirectionalAccuracyLoss, self).__init__()

    def forward(self, predictions, targets):
        """
        Calcula la pérdida de exactitud direccional como 1 - exactitud direccional.
        
        Args:
            predictions (torch.Tensor): Tensor de predicciones.
            targets (torch.Tensor): Tensor de valores reales.
        
        Returns:
            torch.Tensor: Pérdida de exactitud direccional.
        """
        # Compara los signos de las diferencias consecutivas de predicciones y reales
        direction_correct = torch.sign(predictions[1:] - predictions[:-1]) == torch.sign(targets[1:] - targets[:-1])
        # Calcula la exactitud direccional
        directional_accuracy = torch.mean(direction_correct.float())
        # La pérdida es 1 - exactitud direccional (para minimizar la pérdida)
        loss = 1 - directional_accuracy
        return loss
    

if __name__ == "__main__":
    # Simular predicciones y objetivos
    num_steps = 1000
    targets = torch.rand(num_steps) * 2 - 1  # Valores aleatorios entre -1 y 1
    predictions = torch.rand(num_steps) * 2 - 1  # Valores aleatorios entre -1 y 1

    # Inicializar la pérdida de exactitud direccional
    loss_fn = DirectionalAccuracyLoss()

    # Calcular la pérdida en cada paso, mejorando las predicciones gradualmente
    losses = []
    for i in range(num_steps):
        loss = loss_fn(predictions, targets)
        losses.append(loss.item())
        
        # Mejorar las predicciones gradualmente
        predictions = predictions * 0.99 + targets * 0.01

    # Graficar la pérdida a lo largo de los pasos
    plt.plot(losses)
    plt.xlabel('Paso')
    plt.ylabel('Pérdida de Exactitud Direccional')
    plt.title('Pérdida de Exactitud Direccional a lo Largo de los Pasos')
    plt.show()
