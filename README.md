# J-Conv en Espacios de Pontryagin: Marco Matemático, Definición, Herramientas y Anexos

## 0. Propósito del documento
Documento completo sobre la formulación, análisis y herramientas para implementar **convoluciones indefinidas (J-Conv)** dentro del marco de los **espacios de Pontryagin**. Contiene el marco teórico, la definición operativa, componentes para arquitecturas profundas, ventajas, desventajas, y anexos técnicos con definiciones formales, pseudocódigo y validaciones matemáticas.

---

## 1. Marco Matemático

### 1.1 Espacios de Pontryagin
Un **espacio de Pontryagin** $\Pi_\kappa$ es un espacio vectorial real o complejo con producto interno indefinido $[x,y]_J = \langle Jx, y\rangle$, donde $J = \mathrm{diag}(I_p, -I_q)$ y $q = \kappa < \infty$. La descomposición fundamental es:
\[
\mathcal{H}_J = \mathcal{H}_+ \oplus \mathcal{H}_-, \quad [x,y]_J = \langle x_+, y_+ \rangle - \langle x_-, y_- \rangle.
\]

Las subpartes $\mathcal H_+,\mathcal H_-$ son ortogonales respecto a $[\cdot,\cdot]_J$ y su diferencia de dimensión define el índice $\kappa$.

### 1.2 Operadores J-adjuntos y J-isometrías
Para un operador acotado $A$ sobre $\Pi_\kappa$:
- **J-adjunto:** $A^{\sharp} = J^{-1} A^* J.$
- **J-isometría:** $A^{\sharp} A = I$ (preserva $[x,x]_J$).
- **J-unitario:** $A^{\sharp} = A^{-1}.$
- **J-simétrico:** $A^{\sharp} = A.$

Estas propiedades aseguran estabilidad geométrica y simetría espectral, características deseables para operadores de redes neuronales en espacios indefinidos.

### 1.3 Conexión con RKKS
Si un núcleo $K$ no es positivo definido, se obtiene un **espacio de reproducción de Kreĭn (RKKS)**. Cuando el índice negativo es finito, se trata de un espacio de Pontryagin. En ML, esto permite núcleos de la forma $K = K_+ - K_-$ y extiende el teorema del representador.

---

## 2. Convolución Indefinida (J-Conv)

### 2.1 Definición general
Sea la entrada $X = (X_+, X_-)$ y el conjunto de filtros $W = (W_+, W_-)$. La operación convolucional indefinida es:
\[
\mathrm{Conv}_J(X,W) = \mathrm{conv}(X_+,W_+) - \mathrm{conv}(X_-,W_-).
\]

El backprop mantiene el patrón de signos: los gradientes en la rama negativa se invierten.

### 2.2 Atado ortogonal
Para mantener coherencia geométrica:
- **In-tying:** $W_- = R_{in} W_+$, con $R_{in}^\top R_{in} = I$.
- **Out-tying:** $W_- = R_{out} W_+$, con $R_{out}^\top R_{out} = I$.
- **Output-tying:** $Y_- = R Y_+$, donde $R$ mezcla ortogonalmente las activaciones.

Esto aproxima una **J-isometría discreta**, conservando la métrica indefinida localmente.

### 2.3 Regularizaciones
- $\mathcal L_{ortho} = \|W_+^T W_-\|_F^2$
- $\mathcal L_{iso} = (\|W_+\|_F - \|W_-\|_F)^2$
- Penalización de isotropía: controla cancelaciones $[x,x]_J\approx 0$.

---

## 3. Pipeline Profundo en Pontryagin

### 3.1 Capa Lift2Pontryagin
Transforma vectores o tensores euclídeos $x$ en pares $(x_+,x_-)$. Controla el índice $\kappa = q$ del espacio. Puede implementarse como convolución 1×1 por rama.

### 3.2 Bloques básicos J
- **JConv2d:** convolución separada por rama con resta implícita.
- **JBatchNorm2d:** BN separada.
- **Activaciones:** funciones impares (tanh, GELU, leaky).
- **Pooling:** max/avg por rama.

### 3.3 Clasificador J
- **JLinear:** $f(h)=\langle h_+,V_+\rangle - \langle h_-,V_-\rangle$.
- **JClassifier1x1:** capa densa 1×1 para segmentación.

### 3.4 Arquitecturas
- **CNN-J completa:** todas las capas bajo $J$.
- **Híbrida (encoder J, decoder euclídeo):** mantiene sólo el encoder en Pontryagin.
- **Atado ortogonal:** reemplazo sistemático por **JConv-Ortho** con $R$ ortogonal.

### 3.5 Entrenamiento
- Inicialización balanceada (He/Kaiming por rama).
- Regularización con $L_2$, $\mathcal L_{iso}$, $\mathcal L_{ortho}$.
- Monitoreo: $[X,X]_J$, ratio de energía $\|X_+\|/\|X_-\|$.

---

## 4. Ventajas y Desventajas

### Ventajas
1. Preserva la geometría del espacio ($J$-isometría aproximada).
2. Reduce duplicidad de filtros (codifica patrón y anti-patrón).
3. Aumenta expresividad sin duplicar completamente la arquitectura.
4. Regularización estructural (estabilidad y mejor condicionamiento).

### Desventajas
1. Doble memoria si $p=q$.
2. Entrenamiento más complejo (BN separada, activaciones simétricas).
3. Interpretabilidad menor cuando hay cancelaciones fuertes.

---

## 5. Guía Práctica y Evaluación
- Usar encoder J con $q/p < 0.25$ para balance.
- Métricas: Dice, IoU, Recall, distribuciones de $[X,X]_J$.
- Ablaciones: $\kappa$, tipo de atado, tipo de activación, regularización.

---

## 6. Conclusión
El formalismo de Pontryagin permite incorporar métricas indefinidas en redes convolucionales, extendiendo la capacidad representacional y la estabilidad. Las J-Convs son la generalización geométrica de las convoluciones clásicas, y con atados ortogonales pueden implementarse de forma eficiente y teóricamente consistente.

---

## Anexo A. J-adjuntos para convoluciones discretas

### A.1 Operadores por bloques
\[
T_W \begin{bmatrix}x_+\\x_-\end{bmatrix} = \begin{bmatrix}\mathrm{conv}(x_+,W_+)\\\mathrm{conv}(x_-,W_-)\end{bmatrix}
\]
$[
[x,y]_J = \langle x_+,y_+\rangle - \langle x_-,y_-\rangle = \langle Jx,y\rangle.\]

### A.2 J-adjunto
\[
T_W^{\sharp}=J^{-1}T_W^*J = \begin{bmatrix}K_{++}^*&0\\0&K_{--}^*\end{bmatrix}
\]
$J$-simetría: $K_{++}=K_{++}^*, K_{--}=K_{--}^*$.  
$J$-isometría: $K_{++}^*K_{++}=I, K_{--}^*K_{--}=I$.

### A.3 Cruces de ramas
\[
T_W = \begin{bmatrix}K_{++}&K_{+-}\\K_{-+}&K_{--}\end{bmatrix},\quad T_W^{\sharp}=\begin{bmatrix}K_{++}^*&-K_{-+}^*\\-K_{+-}^*&K_{--}^*\end{bmatrix}
\]

### A.4 Atado ortogonal
$W_- = R W_+$ con $R^\top R = I$ $\Rightarrow$ $K_{--}=R K_{++}$, lo que preserva $K_\pm^*K_\pm \approx I$.

---

## Anexo B. Pseudocódigo de J-Conv-Ortho

### B.1 Modo output-tying
```python
y_pos = conv2d(x_pos, W_pos)
R = householder_chain(R_param)
y_neg = R @ y_pos
```
**Ventajas:** estabilidad y bajo costo.

### B.2 Modo in-tying
```python
W_neg = (R_in ⊗ I_{k^2}) · W_pos
y_pos = conv2d(x_pos, W_pos)
y_neg = conv2d(x_neg, W_neg)
```
**Requisito:** C_in^+ = C_in^-.

### B.3 Regularizaciones
$\mathcal L_{ortho} = \|\mathrm{flatten}(W_+)^T\mathrm{flatten}(W_-)\|_F^2$,  
$\mathcal L_{iso} = (\|\cdot\|_+ - \|\cdot\|_-)^2$.

---

## Anexo C. Integración en Arquitecturas
1. Lift2Pontryagin → JConv-Ortho → JBN → activación simétrica.  
2. Clasificador JLinear o proyección 1×1 (sub/concat).  
3. Regularizaciones y monitoreo geométrico.  
4. Elección de modo de atado según costos y estabilidad.

---

## Anexo D. Validación matemática
- La forma bloque-diagonal con atado ortogonal aproxima operadores **J-isométricos**.  
- BN por rama + atado ortogonal ≈ preservación de $[x,x]_J$.  
- Cumple $K_{+-}=-K_{-+}^*$ si el operador es J-simétrico.  

---

## Anexo E. Decisiones prácticas
- Encoder J + decoder euclídeo para reducir costo.  
- CNN-J completa para máxima coherencia geométrica.  
- Índice $\kappa$ pequeño en datos simples; mayor si existen contrastes fuertes.

