# Hoja de trabajo #2 — Transformers y mecanismos de atención

**CC3092 · Deep Learning y Sistemas Inteligentes** — Universidad del Valle de Guatemala
**Nicolás Concuá**

Contenido del notebook `notebook/hoja2.ipynb`:

1. **Investigación** (secciones 2–4): de seq2seq con RNN a la atención de Bahdanau/Luong, Q-K-V, arquitectura
   Transformer (√d_k, multi-head, codificación posicional, Post-LN vs Pre-LN, complejidad) y las APIs de PyTorch y
   Hugging Face para atención.
2. **Atención y multi-head attention desde cero** en PyTorch.
3. **Escalamiento √d_k** (varianza de q·k, saturación de la softmax, norma del Jacobiano) y **costo computacional**
   (tiempo y memoria vs. n, implementación ingenua vs. `F.scaled_dot_product_attention`).
4. **BERT y GPT-2**: mapas de atención por cabeza con detección automática de patrones, correferencia de «it»
   (*tired* vs *wide*), matriz causal y *attention sinks*, entropía por capa, ablación/poda de cabezas, atención vs.
   gradiente × entrada y vista interactiva con bertviz.
5. **Verificaciones** de la implementación propia contra `F.scaled_dot_product_attention`, `nn.MultiheadAttention`,
   máscaras y `TransformerEncoder` causal.
6. Discusión, conclusiones y referencias.

## Resultados principales

| Experimento | Resultado |
|---|---|
| Var(q·k) con q, k ~ N(0, I) | ≈ d_k (1 → 1024); con /√d_k ≈ 1 |
| Softmax sin escalar, d_k = 64 | 84 % de la masa en una llave (escalada: 24 %) |
| Costo de la atención (MPS) | pendiente log-log ≈ 2; 4 GB de matriz n×n con n = 8192 (8 cabezas, 1 capa) |
| Verificaciones contra PyTorch | 13 / 13, error máx. < 10⁻⁶ |
| Cabezas de BERT que resuelven «it» | 22 / 144 (mejor: capa 7, cabeza 11) |
| GPT-2: atención al primer token | 24 % (capa 1) → ~82 % (capas 10–11); 103 / 144 cabezas > 50 % |
| Poda de cabezas en BERT | apagar 42 % de las cabezas: pérdida MLM 1.78 → 1.88 |
| Atención vs. gradiente × entrada (GPT-2) | Spearman ρ ≈ 0.17 |

## Estructura

```
Hoja_trabajo_2_DL/
├── notebook/
│   └── hoja2.ipynb       # notebook completo y ejecutado
└── reports/              # figuras (.png) y tablas (.csv) generadas por el notebook; informe en PDF
```

## Cómo reproducir

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install torch transformers matplotlib pandas numpy bertviz jupyter
cd notebook
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 hoja2.ipynb
```

`bert-base-uncased` y `gpt2` se descargan automáticamente de Hugging Face. El notebook corre en pocos minutos en CPU;
el benchmark de costo usa CUDA o MPS (Apple Silicon) si están disponibles. Hardware usado: Apple M4 Pro.
