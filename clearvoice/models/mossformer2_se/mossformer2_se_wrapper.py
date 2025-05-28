from .mossformer2 import MossFormer_MaskNet
import torch.nn as nn

class MossFormer2_SE_48K(nn.Module):
    """
    The MossFormer2_SE_48K model for speech enhancement.

    This class encapsulates the functionality of the MossFormer MaskNet
    within a higher-level model. It processes input audio data to produce
    enhanced outputs and corresponding masks.

    Arguments
    ---------
    args : Namespace
        Configuration arguments that may include hyperparameters 
        and model settings (not utilized in this implementation but 
        can be extended for flexibility).

    Example
    ---------
    >>> model = MossFormer2_SE_48K(args).model
    >>> x = torch.randn(10, 180, 2000)  # Example input
    >>> outputs, mask = model(x)  # Forward pass
    >>> outputs.shape, mask.shape  # Check output shapes
    """

    def __init__(self, args):
        super(MossFormer2_SE_48K, self).__init__()
        # Initialize the TestNet model, which contains the MossFormer MaskNet
        self.model = TestNet()  # Instance of TestNet

    def forward(self, x):
        """
        Forward pass through the model.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, S], where B is the batch size,
            N is the number of channels (180 in this case), and S is the
            sequence length (e.g., time frames).

        Returns
        -------
        outputs : torch.Tensor
            Enhanced audio output tensor from the model.

        mask : torch.Tensor
            Mask tensor predicted by the model for speech separation.
        """
        outputs, mask = self.model(x)  # Get outputs and mask from TestNet
        return outputs, mask  # Return the outputs and mask


class TestNet(nn.Module):
    """
    The TestNet class for testing the MossFormer MaskNet implementation.

    This class builds a model that integrates the MossFormer_MaskNet
    for processing input audio and generating masks for source separation.

    Arguments
    ---------
    n_layers : int
        The number of layers in the model. It determines the depth
        of the model architecture, we leave this para unused at this moment.
    """

    def __init__(self, n_layers=18):
        super(TestNet, self).__init__()
        self.n_layers = n_layers  # Set the number of layers
        # Initialize the MossFormer MaskNet with specified input and output channels
        self.mossformer = MossFormer_MaskNet(in_channels=180, out_channels=512, out_channels_final=961)

    def forward(self, input):
        """
        Forward pass through the TestNet model.

        Arguments
        ---------
        input : torch.Tensor
            Input tensor of dimension [B, N, S], where B is the batch size,
            N is the number of input channels (180), and S is the sequence length.

        Returns
        -------
        out_list : list
            List containing the mask tensor predicted by the MossFormer_MaskNet.
        """
        out_list = []  # Initialize output list to store outputs
        # Transpose input to match expected shape for MaskNet
        x = input.transpose(1, 2)  # Change shape from [B, N, S] to [B, S, N]
        
        # Get the mask from the MossFormer MaskNet
        mask = self.mossformer(x)  # Forward pass through the MossFormer_MaskNet
        out_list.append(mask)  # Append the mask to the output list

        return out_list  # Return the list containing the mask
