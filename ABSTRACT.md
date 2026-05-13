## Abstract

**Title:** Parametric Memory Gate (PMG): A Trainable Gating Activation Function with Asymmetric Noise Suppression for High-Volatility Sequence Modeling

**Authors:** ([Алексей Бабанов / pwrmind](https://github.com/pwrmind))

**Keywords:** Activation Functions, Gated Recurrent Neural Networks, Time-Series Forecasting, Noise Suppression, Dynamic Parameterization.

### Content:
Gated recurrent architectures, such as Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU), fundamentally rely on standard sigmoid activation functions to regulate information flow and maintain long-term memory. However, the static exponential nature ($e^x$) of the traditional sigmoid curve introduces persistent numerical noise and lacks data-dependent geometric adaptability, leading to severe degradation in forecasting performance when applied to highly volatile, noisy temporal sequences. 

This paper introduces the **Parametric Memory Gate (PMG)**, a novel, fully differentiable activation layer designed explicitly for sequential gating mechanisms and linear attention systems. PMG reformulates the traditional gate by introducing two learned parameters—the exponential `base` and horizontal `shift`—optimized directly via gradient descent. Mathematically formulated as $f(x) = base^{(x+shift)} / (1 + base^{(x+shift)})$, the function scales its parameters in log-space to enforce strict stability boundaries ($base > 1.0$) and guard against boundary discontinuities. 

Empirical benchmarks demonstrate that PMG possesses two critical architectural advantages: 
1. **Asymmetric Noise Suppression:** PMG decays to near-zero values on the negative domain significantly faster than a standard sigmoid, acting as a non-linear low-pass filter that aggressively dampens chaotic background fluctuations.
2. **Trainable Thresholding:** The parameterizable shift dynamically calibrates the exact activation energy required to update or overwrite memory states based on dataset-specific volatility characteristics.

To evaluate its real-world efficacy, a multi-gate PMG-driven recurrent network was stress-tested against industry-standard PyTorch LSTM and GRU layers on high-volatility financial and cryptocurrency asset sequences (AAPL and BTC-USD daily closure streams). Out-of-sample benchmark results demonstrate that the PMG network drastically minimizes prediction errors, outperforming the native GRU layer by a **3.3x reduction in Test MSE** and the native LSTM layer by a **7.9x reduction in Test MSE**. Beyond financial modeling, the unique mathematical properties of PMG offer highly viable optimization vectors for context-retention in Large Language Models (LLMs), acoustic signal denoising, and sensor fusion applications within edge robotics.
