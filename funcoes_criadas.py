import numpy as np
import matplotlib.pyplot as plt

def Codific(arr_signal_quantized, R):
    b = 2 ** np.arange(R - 1, -1, -1)
    arr_signal_quantized = np.vstack(arr_signal_quantized) // b
    arr_binary = np.bitwise_and(1, arr_signal_quantized)
    return np.ravel(arr_binary)
    
def Descodific(arr_binary, R):
    arr_binary = arr_binary.reshape(arr_binary.size // R, R)
    arr_signal = arr_binary * 2 ** np.arange(R - 1,-1,-1)
    arr_signal = np.sum(arr_signal, axis=1)
    return arr_signal
    
def quantific(R, Vmax, Qtype):
    L = 2 ** R
    d = 2.0 * Vmax / L

    Qtype.lower()
    if Qtype == 'midrise':
        vq = np.arange(-Vmax + d / 2, Vmax + d / 2, d)
        vd = np.arange(-Vmax + d, Vmax , d)
        return vq, vd
    elif Qtype == 'midtread':
        vq = np.arange(-Vmax, Vmax, d)
        vd = np.arange(-Vmax + d / 2, Vmax - d / 2, d)
        return vq, vd
    else:
        print('Nao definiu o tipo de quantificador corretamente')


def Quantificador(x, Vq, Vd):
    x = np.ravel(np.array(x))
    xq = np.zeros((x.size, 1)) 
    x = x[:, np.newaxis]
    b = np.count_nonzero(Vd < x, axis=1).reshape(x.size, 1)
    xq[:] = Vq[b]
    return np.ravel(xq), np.ravel(b)
    
def Measure_SNRp(x, y):
    Px = np.sum(x ** 2)
    Pe = np.sum((x - y) ** 2)
    SNRp = 10 * np.log10(Px / Pe)
    return SNRp

def Measure_SNRt(R, Vmax, x):
    Px = np.sum(x ** 2) / x.size
    SNRt = 6 * R + 10 * np.log10((3 * Px) / (Vmax ** 2))
    return SNRt
    
# Matriz geradora
G = np.array([[1, 0, 0, 0, 1, 1, 1],
              [0, 1, 0, 0, 1, 1, 0],
              [0, 0, 1, 0, 1, 0, 1],
              [0, 0, 0, 1, 0, 1, 1]])

# Matriz de Verificação de Paridade
HT = np.array([[1, 1, 1],
               [1, 1, 0],
               [1, 0, 1],
               [0, 1, 1],
               [1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])

def Hamming_7_4(arr_bits):
    add_zeros = (4 - (arr_bits.size % 4)) % 4
    arr = np.concatenate((arr_bits, np.zeros(add_zeros))) 
    arr = np.split(arr, arr.size // 4)
    c = np.dot(arr[:], G) % 2
    return np.ravel(c).astype(int), add_zeros

def Detetor(arr_binary, n_zeros):
    arr = arr_binary.reshape(int(np.ceil(arr_binary.size / 7)), 7)
    s = (np.dot(arr, HT) % 2).astype(int)
    for i in range(s.size // 3):
        if np.sum(s[i]) != 0:
            e = np.sum((s[i] == HT[:]), axis=1)
            bit_errado = np.where(e == 3)[0]
            arr[i][bit_errado] = not arr[i][bit_errado]
    out = arr[:, : -3]
    out = np.ravel(out)
    return out[0 : out.size - n_zeros]

def BER_pratico(arr_bin_tx, arr_bin_rx):   
    ber = np.sum(np.logical_xor(arr_bin_tx, arr_bin_rx)) / arr_bin_tx.size * 100
    return np.round(ber, 2)