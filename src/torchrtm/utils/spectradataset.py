import torch
from torch.utils.data import Dataset


class SpectraDataset(Dataset):
    """Custom Dataset for loading simulated spectra and initial traits."""

    def __init__(self, simulated_spectra, initial_traits):
        """
        Args:
            simulated_spectra (array-like): The simulated spectral data.
            initial_traits (array-like): The associated trait values.
        """
        self.simulated_spectra = torch.tensor(simulated_spectra, dtype=torch.float32)
        self.initial_traits = torch.tensor(initial_traits, dtype=torch.float32)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.simulated_spectra)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (spectrum, traits)
        """
        spectrum = self.simulated_spectra[idx]
        traits = self.initial_traits[idx]
        return spectrum, traits
