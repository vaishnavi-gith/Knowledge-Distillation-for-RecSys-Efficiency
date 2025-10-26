# 💡 Knowledge Distillation for Recommender System Efficiency

## 🧭 Overview

This project explores the application of **Knowledge Distillation (KD)** to **Collaborative Filtering-based Recommendation Systems**.  
The goal is to **compress a high-capacity Teacher model** into a **lightweight Student model**, achieving substantial reductions in **model size** and **inference latency** while maintaining ranking accuracy.

We employ the **Bayesian Personalized Ranking (BPR)** objective and introduce a **custom hybrid loss function** combining traditional BPR loss with a **soft distillation term (MSE over confidence scores)** to transfer knowledge from Teacher to Student.

---

## 🎯 Key Findings & Research Contributions

This study offers critical insights into the challenges of applying KD in high-sparsity recommendation settings:

1. **Efficiency Success:**  
   Achieved **49.5% model compression**, reducing parameters from ≈97.5K to ≈49.5K, with verified improvements in computational efficiency.

2. **Negative Transfer:**  
   The **Distilled Student** underperformed compared to the **Standard Student**:
   - Distilled Student → Recall@10 = **0.0910**  
   - Standard Student → Recall@10 = **0.0970**

3. **Diagnosis — Knowledge Mismatch Problem:**  
   The Teacher’s soft logits were likely **overfit or noisy**, causing **negative knowledge transfer** when applied to the Student’s constrained (32-D) embedding space.  
   → Highlights the need for **specialized distillation strategies** for sparse, ranking-based tasks.

---

## ⚙️ Model Architectures & Implementation

All models are implemented in **PyTorch** and trained with a **custom BPR sampling strategy**.

### Model Specifications

| Model | Architecture | Embedding Size | Parameters | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Teacher** | Matrix Factorization (MF) | 64 | 97,500 | Performance Ceiling |
| **Standard Student** | Matrix Factorization (MF) | 32 | 49,500 | Efficiency Baseline |
| **Distilled Student** | Matrix Factorization (MF) | 32 | 49,500 | KD Target |

---

## 🧩 Custom Knowledge Distillation Loss

The Student model was optimized using a hybrid loss:

$$
\mathcal{L}_{KD} = (1 - \alpha)\mathcal{L}_{\text{Hard}} + \alpha \mathcal{L}_{\text{Soft}}
$$

- **Hard Loss ($\mathcal{L}_{\text{Hard}}$):** Standard **BPR Loss** guiding the student to rank positive items higher than negatives.  
- **Soft Loss ($\mathcal{L}_{\text{Soft}}$):** **MSE** between Teacher and Student confidence scores  
  $(z_T^{pos} - z_T^{neg})$, scaled by $T^2$.

---

## 🧪 Empirical Results (K = 10)

All experiments were conducted on synthetic data.  
Evaluation metrics include **Recall@10**, **NDCG@10**, and **inference latency** (mean ranking time per user).

| Model | Recall@10 | NDCG@10 | Latency (ms) | Compression Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **Teacher** | **0.1400** | **0.0658** | 0.185 | 1.00× |
| **Standard Student** | 0.0970 | 0.0500 | 0.224 | 1.97× |
| **Distilled Student** | 0.0910 | 0.0432 | **0.173** | 1.97× |

**Conclusion:**  
Student models achieved nearly **2× compression**. However, the **Distilled Student** failed to close the accuracy gap, underscoring the **fragility of KD in sparse ranking tasks**.

---

## 📂 Repository Structure

├── knowledge_distillation_reco.ipynb # Main notebook: data simulation, model training, evaluation
├── models.py # MatrixFactorization model definition (conceptual)
└── README.md # Project documentation
