import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================
# 1. Capa Lift2Pontryagin
# =====================================================
class Lift2Pontryagin(nn.Module):
    """
    Capa de levantamiento: transforma un vector/tensor de características euclídeas
    en una representación en un espacio de Pontryagin con índice κ = q.
    
    Dada una entrada x ∈ R^C, genera dos salidas:
    - z_pos ∈ R^p  (rama positiva, producto interno estándar)
    - z_neg ∈ R^q  (rama negativa, producto interno invertido)
    
    Así, el producto indefinido se calcula como:
        [z, z'] = <z_pos, z'_pos> - <z_neg, z'_neg>
    
    Parámetros:
    -----------
    in_features : int
        Número de canales/características de entrada (C).
    out_pos : int
        Número de canales en la rama positiva (p).
    out_neg : int
        Número de canales en la rama negativa (q).
    bias : bool
        Si True, añade sesgo a cada rama.
    """
    def __init__(self, in_features, out_pos, out_neg, bias=True):
        super().__init__()
        self.fc_pos = nn.Linear(in_features, out_pos, bias=bias)
        self.fc_neg = nn.Linear(in_features, out_neg, bias=bias)

    def forward(self, x):
        z_pos = self.fc_pos(x)
        z_neg = self.fc_neg(x)
        return z_pos, z_neg


# =====================================================
# 2. Convolución indefinida (JConv2d)
# =====================================================
class JConv2d(nn.Module):
    """
    Convolución 2D en espacio de Pontryagin (índice κ).
    
    Implementa:
        Conv_J(X, W) = conv(X_pos, W_pos) - conv(X_neg, W_neg)
    
    Donde:
    - X_pos y W_pos son las ramas positivas
    - X_neg y W_neg son las ramas negativas
    
    Parámetros:
    -----------
    in_pos, in_neg : int
        Canales de entrada en ramas positiva y negativa.
    out_pos, out_neg : int
        Canales de salida en ramas positiva y negativa.
    kernel_size, stride, padding, bias : igual a nn.Conv2d.
    """
    def __init__(self, in_pos, in_neg, out_pos, out_neg, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv_pos = nn.Conv2d(in_pos, out_pos, kernel_size, stride, padding, bias=bias)
        self.conv_neg = nn.Conv2d(in_neg, out_neg, kernel_size, stride, padding, bias=bias)

    def forward(self, x_pos, x_neg):
        y_pos = self.conv_pos(x_pos)
        y_neg = self.conv_neg(x_neg)
        return y_pos, y_neg


# =====================================================
# 3. BatchNorm separada (JBatchNorm2d)
# =====================================================
class JBatchNorm2d(nn.Module):
    """
    Normalización por lotes en Pontryagin: se aplica separadamente
    en ramas positiva y negativa.
    
    Parámetros:
    -----------
    c_pos, c_neg : int
        Número de canales de entrada en ramas positiva y negativa.
    """
    def __init__(self, c_pos, c_neg):
        super().__init__()
        self.bn_pos = nn.BatchNorm2d(c_pos)
        self.bn_neg = nn.BatchNorm2d(c_neg)

    def forward(self, x_pos, x_neg):
        return self.bn_pos(x_pos), self.bn_neg(x_neg)


# =====================================================
# 4. Clasificador lineal indefinido (JLinear)
# =====================================================
class JLinear(nn.Module):
    """
    Capa lineal en espacio de Pontryagin.
    
    Implementa:
        f(x) = <x_pos, W_pos> - <x_neg, W_neg>
    
    Es la análoga a nn.Linear, pero separando ramas positiva y negativa.
    
    Parámetros:
    -----------
    in_pos, in_neg : int
        Número de características de entrada en ramas positiva y negativa.
    out_features : int
        Dimensión de salida (e.g., número de clases).
    bias : bool
        Si True, añade sesgo.
    """
    def __init__(self, in_pos, in_neg, out_features, bias=True):
        super().__init__()
        self.fc_pos = nn.Linear(in_pos, out_features, bias=bias)
        self.fc_neg = nn.Linear(in_neg, out_features, bias=False)  # bias compartido con rama positiva

    def forward(self, x_pos, x_neg):
        return self.fc_pos(x_pos) - self.fc_neg(x_neg)

# ---------- Lift 2D: Euclídeo -> Pontryagin (por píxel) ----------
class JLift2d(nn.Module):
    """
    Levanta un tensor imagen X∈R^{N,C,H,W} a (X_pos, X_neg) con C+ y C- canales.
    Usa conv(1x1) separadas por rama.
    """
    def __init__(self, in_ch, out_pos, out_neg, bias=True):
        super().__init__()
        self.lift_pos = nn.Conv2d(in_ch, out_pos, kernel_size=1, bias=bias)
        self.lift_neg = nn.Conv2d(in_ch, out_neg, kernel_size=1, bias=bias)
    def forward(self, x):
        return self.lift_pos(x), self.lift_neg(x)

# ---------- Proyección Pontryagin -> Euclídeo ----------

class JProject2Euclid(nn.Module):
    """
    Proyecta (x_pos, x_neg) a un tensor euclídeo Y ∈ R^{N, C_euc, H, W}.
    Dos opciones:
      - 'sub':   Y = conv1x1_pos(x_pos) - conv1x1_neg(x_neg)   (interpreta [·,·])
      - 'concat':Y = conv1x1( cat(x_pos, x_neg) )              (aprende mezcla)
    """
    def __init__(self, in_pos, in_neg, out_ch, mode='sub', bias=True):
        super().__init__()
        self.mode = mode
        if mode == 'sub':
            self.head_pos = nn.Conv2d(in_pos, out_ch, kernel_size=1, bias=bias)
            self.head_neg = nn.Conv2d(in_neg, out_ch, kernel_size=1, bias=False)
        elif mode == 'concat':
            self.mix = nn.Conv2d(in_pos + in_neg, out_ch, kernel_size=1, bias=bias)
        else:
            raise ValueError("mode debe ser 'sub' o 'concat'")
    def forward(self, x_pos, x_neg):
        if self.mode == 'sub':
            return self.head_pos(x_pos) - self.head_neg(x_neg)
        else:
            x = torch.cat([x_pos, x_neg], dim=1)
            return self.mix(x)

# ---------- Activación por rama ----------
class JAct(nn.Module):
    """
    Aplica activación por rama (misma no-linealidad en + y -).
    kind: 'tanh' | 'gelu' | 'leaky_relu'
    """
    def __init__(self, kind='tanh', negative_slope=0.1):
        super().__init__()
        self.kind = kind
        self.neg_slope = negative_slope
        if kind == 'gelu':
            self.act = nn.GELU()
        elif kind == 'tanh':
            self.act = nn.Tanh()
        elif kind == 'leaky_relu':
            self.act = nn.LeakyReLU(negative_slope)
        else:
            raise ValueError("kind debe ser 'tanh' | 'gelu' | 'leaky_relu'")
    def forward(self, x_pos, x_neg):
        return self.act(x_pos), self.act(x_neg)