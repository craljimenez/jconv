from Jconv import JConv2d,JConv2dOrtho, JBatchNorm2d, JAct, JLift2d, JProject2Euclid
import torch.nn as nn
import torch
import torch.nn.functional as F


def dice_score(preds, targets, n_classes, device, smooth=1e-6):
    """Calculates the Dice score for each class."""
    dice_scores = []
    preds = torch.argmax(preds, dim=1)
    for cls in range(n_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = pred_inds.sum().float() + target_inds.sum().float()
        dice_scores.append((2. * intersection + smooth) / (union + smooth))
    return torch.tensor(dice_scores, device=device)


def iou_score(preds, targets, n_classes, device, smooth=1e-6):
    """Calculates the Intersection over Union (IoU) for each class."""
    iou_scores = []
    preds = torch.argmax(preds, dim=1)
    for cls in range(n_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        iou_scores.append((intersection + smooth) / (union + smooth))
    return torch.tensor(iou_scores, device=device)


# ---------- Bloque Conv-BN-Act (dos veces) ----------
class JConvBlock(nn.Module):
    """
    (Conv -> BN -> Act) x 2 por rama.
    """
    def __init__(self, in_pos, in_neg, out_pos, out_neg, k=3, bn=True, act='tanh', orth=False, mode='out'):
        super().__init__()
        self.conv1 = JConv2d(in_pos, in_neg, out_pos, out_neg, kernel_size=k, padding=k//2) if not orth\
                else JConv2dOrtho(in_pos, in_neg, out_pos, out_neg, k=k, p=k//2, bias=False, mode = mode)
        self.bn1   = JBatchNorm2d(out_pos, out_neg) if bn else None
        self.act1  = JAct(act)
        self.conv2 = JConv2d(out_pos, out_neg, out_pos, out_neg, kernel_size=k, padding=k//2) if not orth\
                else JConv2dOrtho(out_pos, out_neg, out_pos, out_neg, k=k, p=k//2, bias=False, mode = mode)
        self.bn2   = JBatchNorm2d(out_pos, out_neg) if bn else None
        self.act2  = JAct(act)

    def forward(self, x_pos, x_neg):
        y_pos, y_neg = self.conv1(x_pos, x_neg)
        if self.bn1: y_pos, y_neg = self.bn1(y_pos, y_neg)
        y_pos, y_neg = self.act1(y_pos, y_neg)
        y_pos, y_neg = self.conv2(y_pos, y_neg)
        if self.bn2: y_pos, y_neg = self.bn2(y_pos, y_neg)
        y_pos, y_neg = self.act2(y_pos, y_neg)
        return y_pos, y_neg

# ---------- Downsample (MaxPool) ----------
class JDown(nn.Module):
    def __init__(self, in_pos, in_neg, out_pos, out_neg, k=3, bn=True, act='tanh', orth=False, mode='out'):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = JConvBlock(in_pos, in_neg, out_pos, out_neg, k=k, bn=bn, act=act, orth=orth, mode=mode)
    def forward(self, x_pos, x_neg):
        x_pos = self.pool(x_pos)
        x_neg = self.pool(x_neg)
        return self.block(x_pos, x_neg)

# ---------- Upsample (ConvTranspose2d + concat de skip) ----------
class JUp(nn.Module):
    def __init__(self, in_pos, in_neg, skip_pos, skip_neg, out_pos, out_neg, k=3, bn=True, act='tanh', bilinear=False, orth=False, mode='out'):
        super().__init__()
        if bilinear:
            self.up_pos = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.up_neg = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.reduce_pos = nn.Conv2d(in_pos, in_pos//2, 1)
            self.reduce_neg = nn.Conv2d(in_neg, in_neg//2, 1)
            in_pos_eff = in_pos//2 + skip_pos
            in_neg_eff = in_neg//2 + skip_neg
        else:
            self.up_pos = nn.ConvTranspose2d(in_pos, in_pos//2, kernel_size=2, stride=2)
            self.up_neg = nn.ConvTranspose2d(in_neg, in_neg//2, kernel_size=2, stride=2)
            in_pos_eff = in_pos//2 + skip_pos
            in_neg_eff = in_neg//2 + skip_neg
            self.reduce_pos = None
            self.reduce_neg = None
        self.block = JConvBlock(in_pos_eff, in_neg_eff, out_pos, out_neg, k=k, bn=bn, act=act, orth=orth, mode=mode)

    def forward(self, x_pos, x_neg, skip_pos, skip_neg):
        x_pos = self.up_pos(x_pos)
        x_neg = self.up_neg(x_neg)
        if self.reduce_pos is not None:
            x_pos = self.reduce_pos(x_pos)
            x_neg = self.reduce_neg(x_neg)
        # Concatenación canal: ( + ) con ( + ), ( - ) con ( - )
        x_pos = torch.cat([x_pos, skip_pos], dim=1)
        x_neg = torch.cat([x_neg, skip_neg], dim=1)
        return self.block(x_pos, x_neg)

# ---------- Clasificador 1x1 por rama (mapa de logits) ----------
class JClassifier1x1(nn.Module):
    """
    Produce logits por píxel: logits = conv1x1_pos(x_pos) - conv1x1_neg(x_neg)
    """
    def __init__(self, in_pos, in_neg, n_classes, bias=True):
        super().__init__()
        self.head_pos = nn.Conv2d(in_pos, n_classes, kernel_size=1, bias=bias)
        self.head_neg = nn.Conv2d(in_neg, n_classes, kernel_size=1, bias=False)
    def forward(self, x_pos, x_neg):
        return self.head_pos(x_pos) - self.head_neg(x_neg)
    

class UNetHybridJEnc(nn.Module):
    """
    U-Net híbrida: Encoder en Pontryagin, proyección a Euclídeo, Decoder normal.
    """
    def __init__(self, in_ch=3, base_pos=32, base_neg=8, depth=4, 
                 proj_mode='sub', dec_base=64, n_classes=2, k=3, bn=True, act='tanh', orth=False, mode='out'):
        super().__init__()
        # Encoder J
        self.lift = JLift2d(in_ch, base_pos, base_neg)
        enc = nn.ModuleList()
        chp, chn = base_pos, base_neg
        for d in range(depth):
            outp = chp*2 if d>0 else chp
            outn = chn*2 if d>0 else chn
            if d == 0:
                enc.append(JConvBlock(chp, chn, outp, outn, k=k, bn=bn, act=act, orth=orth, mode=mode)) # type: ignore
            else:
                enc.append(JDown(chp, chn, outp, outn, k=k, bn=bn, act=act, orth=orth, mode=mode)) # type: ignore
            chp, chn = outp, outn
        self.encoder = enc

        # Proyección a Euclídeo
        self.proj = JProject2Euclid(chp, chn, out_ch=dec_base, mode=proj_mode)

        # Decoder euclídeo tipo U-Net (simple y estándar)
        self.up_blocks = nn.ModuleList()
        # Canales de skip en Euclídeo: proyectamos también skips (ligero coste)
        self.skip_projs = nn.ModuleList()
        # recolectar canales J de skips
        skip_p, skip_n = [], []
        chp_tmp, chn_tmp = base_pos, base_neg
        for d in range(depth):
            outp = chp_tmp*2 if d>0 else chp_tmp
            outn = chn_tmp*2 if d>0 else chn_tmp
            skip_p.append(outp); skip_n.append(outn)
            chp_tmp, chn_tmp = outp, outn

        # proyecciones de skips a euclídeo
        for d in range(depth):
            self.skip_projs.append(JProject2Euclid(skip_p[d], skip_n[d], out_ch=dec_base*(2**d)//(2**(depth-1)), mode=proj_mode))

        # construir decoder (transposed conv + convs)
        ch = dec_base
        self.upconvs = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        for d in reversed(range(depth)):
            # upsample
            self.upconvs.append(nn.ConvTranspose2d(ch, ch//2 if ch>16 else ch, 2, 2))
            ch_up = ch//2 if ch>16 else ch
            # concat con skip proyectado
            skip_ch = dec_base*(2**d)//(2**(depth-1))
            self.dec_convs.append(
                nn.Sequential(
                    nn.Conv2d(ch_up + skip_ch, ch_up, 3, padding=1),
                    nn.BatchNorm2d(ch_up),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch_up, ch_up, 3, padding=1),
                    nn.BatchNorm2d(ch_up),
                    nn.ReLU(inplace=True),
                )
            )
            ch = ch_up

        # Cabeza final
        self.head = nn.Conv2d(ch, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder J con almacenamiento de skips
        x_pos, x_neg = self.lift(x)
        skips_p, skips_n = [], []
        for block in self.encoder:
            x_pos, x_neg = block(x_pos, x_neg)
            skips_p.append(x_pos); skips_n.append(x_neg)

        # Bottleneck proyectado a euclídeo
        y = self.proj(x_pos, x_neg)

        # Decoder euclídeo: upsample + concat skip proyectado + convs
        for i, (up, dec) in enumerate(zip(self.upconvs, self.dec_convs)):
            y = up(y)
            sp = self.skip_projs[-1-i](skips_p[-1-i], skips_n[-1-i])
            # ajustar tamaño si hay off-by-one
            if y.shape[-2:] != sp.shape[-2:]:
                y = F.interpolate(y, size=sp.shape[-2:], mode='bilinear', align_corners=False)
            y = torch.cat([y, sp], dim=1)
            y = dec(y)

        logits = self.head(y)
        # Final upsampling to match input image size
        upsample = nn.Upsample(size=x.shape[-2:], mode='bilinear', align_corners=False)
        return upsample(logits)

def build_unet_hybrid_jenc(in_ch=3, base_pos=32, base_neg=8, depth=4,
                           proj_mode='sub', dec_base=64, n_classes=2, k=3, bn=True, act='tanh', orth=False, mode='out'):
    """
    Construye U-Net híbrida: encoder J, decoder euclídeo.
    - proj_mode: 'sub' (Y=pos-neg) o 'concat' (mezcla aprendida 1x1)
    - dec_base: canales euclídeos en el bottleneck del decoder
    """
    return UNetHybridJEnc(in_ch, base_pos, base_neg, depth, proj_mode, dec_base, n_classes, k, bn, act, orth, mode)\
                if not orth \
                    else UNetHybridJEnc(in_ch, base_pos, base_pos, depth, proj_mode, dec_base, n_classes, k, bn, act, orth, mode)

#**********************************************************
# FCN-Híbrida con Encoder J y Decoder simple (bilinear)
#***********************************************************

class FCNHybridJEnc(nn.Module):
    """
    FCN híbrida: Encoder en Pontryagin, proyección a Euclídeo, cabeza 1x1 y upsample bilinear.
    """
    def __init__(self, in_ch=3, base_pos=32, base_neg=8, stages=4, 
                 proj_mode='sub', euc_ch=128, n_classes=21, k=3, bn=True, act='tanh'):
        super().__init__()
        self.lift = JLift2d(in_ch, base_pos, base_neg)
        layers = []
        chp, chn = base_pos, base_neg
        for s in range(stages):
            outp = chp*2 if s>0 else chp
            outn = chn*2 if s>0 else chn
            if s == 0:
                layers.append(JConvBlock(chp, chn, outp, outn, k=k, bn=bn, act=act))
            else:
                layers.append(JDown(chp, chn, outp, outn, k=k, bn=bn, act=act))
            chp, chn = outp, outn
        self.encoder = nn.ModuleList(layers)

        # Proyección a euclídeo y cabeza
        self.proj = JProject2Euclid(chp, chn, out_ch=euc_ch, mode=proj_mode)
        self.head = nn.Conv2d(euc_ch, n_classes, kernel_size=1)

    def forward(self, x):
        initial_size = x.shape[-2:]
        x_pos, x_neg = self.lift(x)
        for block in self.encoder:
            x_pos, x_neg = block(x_pos, x_neg)
        y = self.proj(x_pos, x_neg)       # euclídeo
        logits = self.head(y)
        return F.interpolate(logits, size=initial_size, mode='bilinear', align_corners=False)

def build_fcn_hybrid_jenc(in_ch=3, base_pos=32, base_neg=8, stages=4,
                          proj_mode='sub', euc_ch=128, n_classes=21, k=3, bn=True, act='tanh'):
    """
    Construye FCN híbrida: encoder J, decoder simple bilinear.
    - proj_mode: 'sub' (pos-neg) o 'concat' (mezcla aprendida 1x1)
    - euc_ch: canales euclídeos tras proyección
    """
    return FCNHybridJEnc(in_ch, base_pos, base_neg, stages, proj_mode, euc_ch, n_classes, k, bn, act)


#############################################################
# UNET y FCN art-state
#############################################################

# ----- Bloque Conv-BN-ReLU (x2) -----
class ConvBlock(nn.Module):
    """
    Bloque clásico: (Conv2d -> BN -> ReLU) x 2
    """
    def __init__(self, in_ch, out_ch, k=3, bn=True):
        super().__init__()
        layers = []
        for i in range(2):
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size=k, padding=k//2))
            if bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ----- Downsample: MaxPool + ConvBlock -----
class Down(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, bn=True):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch, k=k, bn=bn)

    def forward(self, x):
        return self.conv(self.pool(x))


# ----- Upsample: ConvTranspose + concat + ConvBlock -----
class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, k=3, bn=True, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.reduce = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1)
            in_eff = in_ch // 2 + skip_ch
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            in_eff = in_ch // 2 + skip_ch
            self.reduce = None

        self.conv = ConvBlock(in_eff, out_ch, k=k, bn=bn)

    def forward(self, x, skip):
        x = self.up(x)
        if self.reduce is not None:
            x = self.reduce(x)
        # Ajuste de tamaños si no coinciden exactamente
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
    
class UNet(nn.Module):
    """
    U-Net clásica completamente euclídea.
    Arquitectura:
      encoder: ConvBlock -> Down x (depth-1)
      bottleneck: ConvBlock
      decoder: Up x (depth-1)
      head: Conv1x1 a n_classes
    """
    def __init__(self, in_ch=3, base_ch=64, depth=4, n_classes=2, k=3, bn=True, bilinear=False):
        super().__init__()
        # Encoder
        self.inc = ConvBlock(in_ch, base_ch, k=k, bn=bn)
        self.downs = nn.ModuleList()
        ch = base_ch
        for _ in range(1, depth):
            self.downs.append(Down(ch, ch * 2, k=k, bn=bn))
            ch *= 2

        # Bottleneck
        self.bottleneck = ConvBlock(ch, ch * 2, k=k, bn=bn)
        bott_ch = ch * 2

        # Decoder
        self.ups = nn.ModuleList()
        dec_ch = bott_ch
        for i in range(depth - 1):
            skip_ch = ch // (2 ** (i + 1))
            self.ups.append(Up(dec_ch, skip_ch, dec_ch // 2, k=k, bn=bn, bilinear=bilinear))
            dec_ch //= 2

        # Cabeza de clasificación
        self.head = nn.Conv2d(dec_ch, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder con skips
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i, up in enumerate(self.ups):
            skip = skips[-(i + 2)]  # recupera skip del mismo nivel
            x = up(x, skip)

        return self.head(x)


def build_unet(in_ch=3, base_ch=64, depth=4, n_classes=2, k=3, bn=True, bilinear=False):
    """
    Crea una U-Net tradicional.
    Parámetros:
      - in_ch: canales de entrada (RGB=3)
      - base_ch: canales base del encoder
      - depth: nº de niveles encoder/decoder
      - n_classes: nº de clases de salida
      - k: tamaño del kernel conv
      - bn: BatchNorm
      - bilinear: upsampling bilineal o ConvTranspose
    """
    return UNet(in_ch, base_ch, depth, n_classes, k, bn, bilinear)

class FCN(nn.Module):
    """
    Fully Convolutional Network clásica (tipo FCN-32).
    Encoder convolucional con downsampling progresivo + cabeza 1x1 + upsample bilinear.
    """
    def __init__(self, in_ch=3, base_ch=64, stages=4, n_classes=21, k=3, bn=True):
        super().__init__()
        layers = []
        ch = base_ch
        layers.append(ConvBlock(in_ch, ch, k=k, bn=bn))
        for s in range(1, stages):
            layers.append(Down(ch, ch * 2, k=k, bn=bn))
            ch *= 2
        self.encoder = nn.ModuleList(layers)
        self.head = nn.Conv2d(ch, n_classes, kernel_size=1)
        self.up_factor = 2 ** (stages - 1)

    def forward(self, x):
        initial_size = x.shape[-2:]
        for block in self.encoder:
            x = block(x)
        logits = self.head(x)
        # Upsample al tamaño de entrada
        return F.interpolate(logits, size=initial_size, mode='bilinear', align_corners=False)


def build_fcn(in_ch=3, base_ch=64, stages=4, n_classes=21, k=3, bn=True):
    """
    Crea una FCN tradicional.
    Parámetros:
      - in_ch: canales de entrada (RGB=3)
      - base_ch: canales iniciales del encoder
      - stages: nº de bloques de downsample
      - n_classes: nº de clases
      - k: kernel convolucional
      - bn: usar BatchNorm
    """
    return FCN(in_ch, base_ch, stages, n_classes, k, bn)