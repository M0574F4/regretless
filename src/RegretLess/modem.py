import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# --------------------------------------------------
#     RRC Filter Design (like Sionna get_psf)
# --------------------------------------------------
def design_rrc_torch(sps, span, beta):
    """
    Returns a 1D RRC filter of shape (1,1,K) for conv1d with 'same' padding.
    """
    # Number of taps
    num_taps = sps * span
    t = np.arange(num_taps) - (num_taps - 1) / 2  # center at zero

    # Convert to "symbol time"
    t_symbol = t / sps
    # Standard RRC formula
    # Avoid divide-by-zero with np.errstate
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = (np.sin(np.pi * t_symbol * (1 - beta)) +
                     4 * beta * t_symbol * np.cos(np.pi * t_symbol * (1 + beta)))
        denominator = (np.pi * t_symbol * (1 - (4 * beta * t_symbol) ** 2))
        rrc = numerator / denominator

    # Fix the singularity at t=0
    rrc[np.isnan(rrc)] = 1.0
    # Normalize
    energy = np.sum(rrc ** 2)
    if energy > 1e-15:
        rrc /= np.sqrt(energy)
    # Convert to torch and add dimensions for conv1d
    rrc_t = torch.tensor(rrc, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return rrc_t

def conv1d_same(x, kernel):
    """
    Mimic TF 'same' padding for conv1d with stride=1.
    x: (B,1,L)
    kernel: (1,1,K)
    output: (B,L)
    """
    K = kernel.shape[-1]
    pad_left = (K - 1) // 2
    pad_right = K - 1 - pad_left
    x_padded = F.pad(x, (pad_left, pad_right))
    y = F.conv1d(x_padded, kernel, stride=1)
    return y.squeeze(1)

# --------------------------------------------------
#  QPSK Modulator (Mimic Sionna Approach in PyTorch)
# --------------------------------------------------
class QPSKModulator(nn.Module):
    def __init__(self, sps=16, span=8, beta=0.5):
        super().__init__()
        self.sps = sps
        self.span = span
        self.beta = beta
        self.rrc = design_rrc_torch(sps, span, beta)
        # QPSK mapping dictionary
        self.map_dict = {
            (0, 0): (1 + 1j) / math.sqrt(2),
            (0, 1): (-1 + 1j) / math.sqrt(2),
            (1, 1): (-1 - 1j) / math.sqrt(2),
            (1, 0): (1 - 1j) / math.sqrt(2),
        }

    def forward(self, bits):
        """
        bits: (B, num_bits), with num_bits even.
        Returns:
            tx_waveform: real tensor of shape (B, L, 2) where the last
                         dimension holds [real, imag].
        """
        B, num_bits = bits.shape
        assert num_bits % 2 == 0, "The number of bits must be even."
        num_symbols = num_bits // 2

        # 1) Map bits -> QPSK symbols
        bits2 = bits.view(B, num_symbols, 2)
        symbols_list = []
        for b_i in range(B):
            row = []
            for s in bits2[b_i]:
                key = (int(s[0].item()), int(s[1].item()))
                row.append(self.map_dict[key])
            symbols_list.append(row)
        symbols_array = np.array(symbols_list, dtype=np.complex64)  # shape: (B, num_symbols)
        symbols_tensor = torch.from_numpy(symbols_array)  # tensor with cfloat

        # 2) Upsample: insert zeros between symbols
        up_len = num_symbols * self.sps
        upsampled = torch.zeros((B, up_len), dtype=torch.cfloat)
        upsampled[:, ::self.sps] = symbols_tensor

        # 3) Half-symbol shift: pad and slice
        offset = self.sps // 2
        up_pad = F.pad(upsampled, (offset, 0))  # pad at the beginning
        up_shifted = up_pad[:, :up_len]

        # 4) Matched filtering (separate conv1d on real & imag parts)
        x_real = up_shifted.real.unsqueeze(1)  # (B, 1, L)
        x_imag = up_shifted.imag.unsqueeze(1)
        tx_r = conv1d_same(x_real, self.rrc)
        tx_i = conv1d_same(x_imag, self.rrc)

        # 5) Multiply by sqrt(sps)
        tx_r = tx_r * np.sqrt(self.sps)
        tx_i = tx_i * np.sqrt(self.sps)

        # 6) Stack real and imaginary parts along the last dimension: (B, L, 2)
        tx_waveform = torch.stack((tx_r, tx_i), dim=-1)
        return tx_waveform

# --------------------------------------------------
#  QPSK Demodulator (Mimic Sionna Approach in PyTorch)
# --------------------------------------------------
class QPSKDemodulator(nn.Module):
    def __init__(self, sps=16, span=8, beta=0.5):
        super().__init__()
        self.sps = sps
        self.span = span
        self.beta = beta
        self.rrc = design_rrc_torch(sps, span, beta)
        
        # Constellation points in the same order as in the modulator
        self.constellation = torch.tensor([1+1j, -1+1j, -1-1j, 1-1j], dtype=torch.cfloat)
        self.bit_map = {
            0: torch.tensor([0, 0]),
            1: torch.tensor([0, 1]),
            2: torch.tensor([1, 1]),
            3: torch.tensor([1, 0]),
        }

    def forward(self, rx_waveform, num_bits):
        """
        rx_waveform: real tensor of shape (B, L, 2) where the last dimension is [real, imag]
        num_bits: original transmitted number of bits.
        Returns:
            recovered_bits: (B, num_bits)
        """
        B, L, two = rx_waveform.shape
        assert two == 2, "The last dimension must hold [real, imag] components."
        num_symbols = num_bits // 2

        # 0) Convert rx_waveform from real-valued to complex-valued representation
        rx_complex = torch.complex(rx_waveform[..., 0], rx_waveform[..., 1])  # shape: (B, L)

        # 1) Matched filter: separate conv1d for real and imag parts
        rx_real = rx_complex.real.unsqueeze(1)  # (B, 1, L)
        rx_imag = rx_complex.imag.unsqueeze(1)
        y_r = conv1d_same(rx_real, self.rrc)
        y_i = conv1d_same(rx_imag, self.rrc)
        y_filtered = torch.complex(y_r, y_i)

        # 2) Divide by sqrt(sps)
        y_filtered = y_filtered / np.sqrt(self.sps)

        # 3) Downsample with offset (sps//2)
        offset = self.sps // 2
        start = offset
        end = offset + num_symbols * self.sps
        end = min(end, y_filtered.size(1))
        y_slice = y_filtered[:, start:end]
        symbols = y_slice[:, ::self.sps]  # (B, num_symbols)
        if symbols.shape[1] < num_symbols:
            pad_count = num_symbols - symbols.shape[1]
            pad_zeros = symbols.new_zeros((B, pad_count), dtype=torch.cfloat)
            symbols = torch.cat([symbols, pad_zeros], dim=1)

        # 4) Make symbol decisions by finding the nearest constellation point
        sym_exp = symbols.unsqueeze(-1)  # (B, num_symbols, 1)
        const_exp = self.constellation.view(1, 1, -1)  # (1, 1, 4)
        dists = torch.abs(sym_exp - const_exp)
        decisions = torch.argmin(dists, dim=-1)  # (B, num_symbols)

        # 5) Map decisions to bits
        bit_lookup = torch.stack([self.bit_map[0], self.bit_map[1],
                                  self.bit_map[2], self.bit_map[3]], dim=0)  # (4, 2)
        bits_out = bit_lookup[decisions]  # (B, num_symbols, 2)
        bits_out = bits_out.view(B, -1)  # (B, num_bits)
        return bits_out

# --------------------------------------------------
#          DEMO
# --------------------------------------------------
if __name__ == "__main__":
    # Parameters
    sps = 32
    span = 8
    beta = 0.5
    B = 2  # batch size
    num_bits = 1000  # should be even

    mod = QPSKModulator(sps, span, beta)
    dem = QPSKDemodulator(sps, span, beta)

    # Generate random bits
    bits_in = torch.randint(0, 2, (B, num_bits))

    # Modulate: output is (B, L, 2) real-valued where last dim is [real, imag]
    tx_waveform = mod(bits_in)
    print("tx_waveform shape:", tx_waveform.shape)

    # (Optional) add channel or noise here; for now, we simply pass through
    rx_waveform = tx_waveform.clone()

    # Demodulate: input is now real-valued with shape (B, L, 2)
    bits_out = dem(rx_waveform, num_bits)
    print("bits_out shape:", bits_out.shape)

    # Compare to compute BER
    n_err = (bits_in != bits_out).sum().item()
    print(f"Errors: {n_err}/{bits_in.numel()} => BER={n_err / bits_in.numel():.3g}")
