import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # مقداردهی وزن‌ها با توزیع نرمال
        self.weights_input_to_hidden = np.random.uniform(
            -1, 1, (input_size, hidden_size)
        )
        self.weights_hidden_to_output = np.random.uniform(
            -1, 1, (hidden_size, output_size)
        )
        self.bias_hidden = np.random.uniform(-1, 1, (1, hidden_size))
        self.bias_output = np.random.uniform(-1, 1, (1, output_size))

    def forward(self, inputs):
        # لایه مخفی
        hidden_layer = np.dot(inputs, self.weights_input_to_hidden) + self.bias_hidden
        hidden_layer = np.tanh(hidden_layer)  # فعال‌سازی تانژانت هیپربولیک

        # لایه خروجی
        output_layer = (
            np.dot(hidden_layer, self.weights_hidden_to_output) + self.bias_output
        )
        output_layer[:, 0] = np.clip(output_layer[:, 0], 0, 1)  # خروجی حرکت به جلو
        output_layer[:, 1] = np.tanh(output_layer[:, 1])  # خروجی چرخش

        return output_layer


# تست شبکه
nn = NeuralNetwork(input_size=3, hidden_size=5, output_size=2)

# ورودی نرمال‌سازی شده
inputs = np.array([[0.5, 0.5, -0.5]])  # مثلاً موقعیت و زاویه

outputs = nn.forward(inputs)
print("Outputs:", outputs)
