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

# =====================================================
# 5JconvOrth: Convolución ortogonal en Pontryagin
# =====================================================

def householder_chain(vs):
    """
    Construye una matriz ortogonal R = H_m ... H_2 H_1 a partir de m vectores columna.
    Cada H(v) = I - 2 (v v^T)/(v^T v). Devuelve R (n x n).
    vs: (n, m) con m reflectores; si m=0 -> I
    """
    n, m = vs.shape
    R = torch.eye(n, device=vs.device, dtype=vs.dtype)
    for j in range(m):
        v = vs[:, j:j+1]  # (n,1)
        denom = torch.clamp(v.t() @ v, min=1e-12)  # estabilidad
        H = torch.eye(n, device=vs.device, dtype=vs.dtype) - 2.0 * (v @ v.t()) / denom
        R = H @ R
    return R

class JConv2dOrtho(nn.Module):
    """
    Convolución 2D indefinida con atado ortogonal entre ramas:
        y_pos = conv(x_pos, W_pos)
        y_neg = conv(x_neg, W_neg),   con   W_neg = R * W_pos
    Modo de atado:
      - mode='in':    R actúa sobre C_in (requiere in_pos == in_neg)
      - mode='out':   R actúa sobre C_out (requiere out_pos == out_neg)
      - mode='output': y_neg = R_out(y_pos) (mezcla ortogonal post-conv)

    Parametrización ortogonal: cadena de reflectores de Householder (estable y sin inversas).
    num_reflectors controla capacidad (p.ej., 1–4 suele bastar).
    """
    def __init__(self, in_pos, in_neg, out_pos, out_neg, k=3, s=1, p=1, bias=True,
                 mode='in', num_reflectors=2):
        super().__init__()
        assert mode in {'in', 'out', 'output'}
        self.mode = mode
        self.conv_pos = nn.Conv2d(in_pos, out_pos, kernel_size=k, stride=s, padding=p, bias=bias)

        if mode == 'in':
            assert in_pos == in_neg, "Para mode='in' se requiere in_pos == in_neg (R cuadrada)."
            self.in_neg = in_neg
            # Parametrizamos R_in con m reflectores sobre R^{C_in}
            self.R_param = nn.Parameter(torch.randn(in_pos, num_reflectors) * 0.01)
        elif mode == 'out':
            assert out_pos == out_neg, "Para mode='out' se requiere out_pos == out_neg (R cuadrada)."
            self.out_neg = out_neg
            self.R_param = nn.Parameter(torch.randn(out_pos, num_reflectors) * 0.01)
        else:  # mode == 'output'
            assert out_pos == out_neg, "Para mode='output' se requiere out_pos == out_neg."
            self.mix = None  # R se construye on-the-fly sobre C_out
            self.R_param = nn.Parameter(torch.randn(out_pos, num_reflectors) * 0.01)

    @property
    def conv_neg(self):
        """
        Expose a module-like handle for the negative branch when weights are tied orthogonally.
        This keeps layer-path resolution compatible with non-orthogonal models.
        """
        return self

    def forward(self, x_pos, x_neg):
        y_pos = self.conv_pos(x_pos)

        if self.mode == 'output':
            # y_neg = R_out y_pos  (mezcla ortogonal de canales de salida)
            B, C, H, W = y_pos.shape
            R = householder_chain(self.R_param)  # (C,C)
            y_pos_flat = y_pos.flatten(2)        # (B,C,H*W)
            y_neg_flat = R @ y_pos_flat          # (B,C,H*W) con broadcast implícito si reordenamos
            y_neg = y_neg_flat.view(B, C, H, W)
            return y_pos, y_neg

        # Para 'in' y 'out' debemos construir W_neg a partir de W_pos
        Wp = self.conv_pos.weight  # (out_pos, in_pos, k, k)
        bp = self.conv_pos.bias

        if self.mode == 'in':
            # R_in actúa sobre eje C_in y preserva kxk
            Cin = Wp.shape[1]
            R = householder_chain(self.R_param)   # (Cin,Cin)
            Wp_mat = Wp.view(Wp.shape[0], Cin, -1)             # (Cout, Cin, k^2)
            Wn_mat = torch.einsum('ij,ocj->oci', R, Wp_mat)    # (Cout, Cin, k^2)
            Wn = Wn_mat.view_as(Wp)                            # (Cout, Cin, k, k)
        else:  # mode == 'out'
            # R_out actúa sobre eje C_out
            Cout = Wp.shape[0]
            R = householder_chain(self.R_param)   # (Cout,Cout)
            Wp_mat = Wp.view(Cout, -1)                          # (Cout, Cin*k^2)
            Wn_mat = R @ Wp_mat                                 # (Cout, Cin*k^2)
            Wn = Wn_mat.view_as(Wp)

        # y_neg = conv(x_neg, W_neg) (sin bias para simetría; opcional añadir bias propio)
        y_neg = F.conv2d(x_neg, Wn, bias=None, stride=self.conv_pos.stride,
                         padding=self.conv_pos.padding, dilation=self.conv_pos.dilation,
                         groups=self.conv_pos.groups)
        return y_pos, y_neg
