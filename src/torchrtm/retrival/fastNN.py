import torch
import torch.nn as nn
import os


class Inverse_Net(nn.Module):
    def __init__(
        self,
        layer_dims=[2001, 6],  # user-defined layer structure
        use_attention=True,
        dropout_rate=0.5
    ):
        """
        Parameters
        ----------
        layer_dims : list[int]
            Example: [2001, 100, 10] means input 2001-D → hidden 100-D → output 10-D.
        use_attention : bool
            Whether to enable the attention-based multi-head mode.
        dropout_rate : float
            Dropout rate used for non-attention MLP layers.
        """
        super(Inverse_Net, self).__init__()

        self.use_attention = use_attention
        self.heads = heads = layer_dims[-1]
        self.flatten = nn.Flatten()

        # ---------- Mode 1: Multi-head attention ----------
        if use_attention:
            # Adjust dimensions for attention mode
            layer_dims[-1] = 1
            layer_dims.insert(0, layer_dims[0])

            input_size = layer_dims[0]
            hidden_size = layer_dims[1] if len(layer_dims) > 1 else input_size
            output_size = layer_dims[-1]

            # Attention layers (one per head)
            self.attention_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_size, hidden_size, bias=False),
                    nn.Sigmoid()
                ) for _ in range(heads)
            ])

            # Output layers for each head
            self.end_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, output_size),
                    nn.Sigmoid()
                ) for _ in range(heads)
            ])

            # Final fusion layer for combining multi-head outputs
            self.final_layer = nn.Sequential(
                nn.Linear(heads, heads),
                nn.Sigmoid()
            )

        # ---------- Mode 2: Standard MLP ----------
        else:
            layers = []
            for i in range(len(layer_dims) - 1):
                layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
                # Add activation and dropout for hidden layers only
                if i < len(layer_dims) - 2:
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(p=dropout_rate))

            self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # ---------- Mode 1: Multi-head attention ----------
        if self.use_attention:
            outputs = []
            for i in range(self.heads):
                # Compute attention weights for each head
                attention_weights = self.attention_layers[i](x)
                weighted_x = torch.multiply(attention_weights, x)

                # Pass through the head-specific output layer
                output = self.end_layers[i](weighted_x)
                outputs.append(output)

            print(output.shape)

            # Concatenate all head outputs
            concatenated_output = self.flatten(torch.cat(outputs, dim=1))
            print(concatenated_output.shape)

            # Compute final attention weights and apply them
            final_attention_weights = self.final_layer(concatenated_output)
            final_output = torch.multiply(final_attention_weights, concatenated_output)
            return final_output

        # ---------- Mode 2: Standard MLP ----------
        else:
            return self.mlp(x)


def load_encoder(weights_path=None, device="cpu"):
    if weights_path is None:
        # 自动获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, "weights", "peng_2025_weights.pt")
        print(weights_path)
    model = Inverse_Net(layer_dims=[2001, 6], use_attention=True)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def normlization_torch_rse2025(x,
                        x_max=[5, 296.4073170731706, 85.05, 7.5, 0.1574634146341464, 0.0471951219512196],
                        x_min=[1, 0, 0, 0, 0, 0],
                        fitting=True):
    """
    Normalizes the input PyTorch tensor `x` based on the given max and min values.
    If `fitting` is True, scales the input to a range of [0, 1], otherwise, applies reverse scaling.

    Args:
    x: Input PyTorch tensor to be normalized.
    x_max: Max value for normalization (converted to torch tensor).
    x_min: Min value for normalization (converted to torch tensor).
    fitting: Boolean to indicate if normalization (True) or reverse scaling (False) is applied.

    Returns:
    Normalized or reverse-scaled PyTorch tensor.
    """
    # Convert x_max and x_min to torch tensors
    x_max = torch.tensor(x_max, dtype=torch.float32)
    x_min = torch.tensor(x_min, dtype=torch.float32)

    try:
        x2 = x.clone()  # For torch tensors
    except AttributeError:
        x2 = x.copy()  # For NumPy arrays (if applicable)

    # Handle 2D or other dimensional inputs
    if len(x2.shape) == 2:
        for i in range(x2.shape[-1]):
            if fitting:
                x2[:, i] = (x[:, i] - x_min[i]) / (x_max[i] - x_min[i])
            else:
                x2[:, i] = x[:, i] * (x_max[i] - x_min[i]) + x_min[i]
    else:
        # Assuming the length is fixed at 6
        for i in range(6):
            if fitting:
                x2[:, i] = (x[:, i] - x_min[i]) / (x_max[i] - x_min[i])
            else:
                x2[:, i] = x[:, i] * (x_max[i] - x_min[i]) + x_min[i]

    return x2
def load_pinns_predict(x_raw, model = None, device="cpu"):

    model = load_encoder()
    model.eval()
    x_raw = torch.tensor(x_raw, dtype=torch.float32).to(device)

    x_norm = normlization_torch_rse2025(x_raw, fitting=False)

    with torch.no_grad():
        y_pred_norm = model(x_norm)

    return y_pred_real.cpu().numpy()