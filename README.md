# RadLite: Multi-Task LoRA Fine-Tuning of Small Language Models for CPU-Deployable Radiology AI

[![arXiv](https://img.shields.io/badge/arXiv-2605.00421-b31b1b.svg)](https://arxiv.org/abs/2605.00421)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1o_DoIIiYVk1OIfh5kliSbbtkO8ES64Ze#scrollTo=t60gcjhOIX4_)

**Pankaj Gupta, MD** and **Kartik Bose**

*Postgraduate Institute of Medical Education and Research, Chandigarh, India*

RadLite demonstrates that small language models (3-4B parameters) can achieve strong multi-task radiology performance through LoRA fine-tuning on 162K samples spanning 9 clinical tasks — and run entirely on consumer CPUs without GPU requirements.

## Key Results

### Zero-Shot vs. Fine-Tuned Performance

| Task | Metric | Qwen2.5-3B ZS | Qwen2.5-3B FT | Qwen3-4B ZS | Qwen3-4B FT | Best Model |
|------|--------|:-:|:-:|:-:|:-:|:-:|
| RADS Assignment | Accuracy | 0.242 | **0.770** | 0.242 | 0.764 | Qwen2.5 |
| Impression Generation | ROUGE-L | 0.132 | **0.502** | 0.135 | 0.274 | Qwen2.5 |
| Temporal Comparison | Jaccard | 0.493 | 0.293 | 0.542 | **0.923** | Qwen3 |
| Radiology NER | ROUGE-L | 0.047 | 0.030 | 0.049 | **0.950** | Qwen3 |
| N-Staging | Accuracy | 0.000 | **0.890** | 0.000 | **0.890** | Tie |
| M-Staging | Accuracy | 0.000 | **0.730** | 0.000 | **0.730** | Tie |
| Radiology NLI | Accuracy | 0.223 | **0.825** | 0.263 | 0.817 | Qwen2.5 |
| Abnormality Detection | Per-label Acc | — | 0.000 | — | **0.606** | Qwen3 |
| Radiology QA | ROUGE-L | 0.035 | **0.107** | 0.027 | 0.093 | Qwen2.5 |

Fine-tuning produces dramatic improvements: RADS accuracy +53pp, N-staging +89pp, NLI +60pp, and NER ROUGE-L +1,840% for Qwen3.

### Complementary Model Strengths

- **Qwen2.5-3B** excels at **generation tasks**: impression generation (+83% over Qwen3), RADS accuracy, NLI, QA
- **Qwen3-4B** excels at **extraction tasks**: NER (0.950 vs 0.030), temporal comparison (0.923 vs 0.293), abnormality detection

The task-routed oracle ensemble (selecting the best model per task) achieves the best performance across all 9 tasks.

### Per-RADS System Accuracy (Fine-Tuned)

| System | Qwen2.5-3B | Qwen3-4B | Test Samples |
|--------|:-:|:-:|:-:|
| VI-RADS | 96.7% | 96.7% | 30 |
| TI-RADS | 85.0% | 85.0% | 80 |
| PI-RADS | 85.0% | 83.3% | 60 |
| BI-RADS | 73.3% | 77.5% | 400 |
| LI-RADS | 72.5% | 72.5% | 80 |
| O-RADS | 70.0% | 70.0% | 30 |
| NI-RADS | 63.3% | 50.0% | 40 |
| Lung-RADS | 50.0% | 62.5% | 8 |
| GB-RADS | 33.3% | 33.3% | 3 |
| CAD-RADS | 40.0% | 26.7% | 15 |

### CPU Deployment Benchmarks

| Model | GGUF Size | Quantization | CPU Speed | RADS Query Time |
|-------|:-:|:-:|:-:|:-:|
| Qwen2.5-3B | 1.8 GB | Q4_K_M | 7.7 tok/s | ~2.6s |
| Qwen3-4B | 2.4 GB | Q4_K_M | 4.4 tok/s | ~4.5s |
| Both (ensemble) | 4.2 GB | Q4_K_M | — | — |

*Benched on Intel i7-class CPU, 4 threads.*

## Model Weights

Quantized GGUF models (Q4_K_M) are available via Google Drive:

| Model | Size | Link |
|-------|------|------|
| Qwen2.5-3B (generation tasks) | 1.8 GB | [Download](https://drive.google.com/file/d/1Y6FOOVW-0y_h3OM5_FjojoxETdyZAbD3/view) |
| Qwen3-4B (extraction tasks) | 2.4 GB | [Download](https://drive.google.com/file/d/1ghZ3DihYLrxClq1euSCalNvTfe8susFp/view) |

## Quick Start

### Option 1: Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1o_DoIIiYVk1OIfh5kliSbbtkO8ES64Ze#scrollTo=t60gcjhOIX4_)

The notebook downloads both models automatically and runs all 9 tasks with a task router that selects the optimal model per task.

### Option 2: Local Inference

```bash
pip install llama-cpp-python gdown
```

```python
from llama_cpp import Llama

# Download models first (or use gdown)
model = Llama("qwen2.5-3b-radiology-Q4_K_M.gguf", n_ctx=2048, n_threads=4, verbose=False)

prompt = (
    "<|im_start|>system\nYou are an expert radiologist.<|im_end|>\n"
    "<|im_start|>user\n[TASK: rads_assignment]\nClassify the following report "
    "into the appropriate RADS category. Output ONLY the RADS category.\n\n"
    "Report:\nBreast ultrasound shows a 2.1 cm irregular hypoechoic mass with angular "
    "margins and posterior shadowing.<|im_end|>\n"
    "<|im_start|>assistant\n"
)

response = model(prompt, max_tokens=30, temperature=0.1, stop=["<|im_end|>"])
print(response["choices"][0]["text"].strip())
# Output: BI-RADS 4
```

## Supported Tasks

| Task | Description | Routed Model | Max Tokens |
|------|-------------|:---:|:---:|
| `rads_assignment` | RADS classification across 10 systems | Qwen2.5-3B | 30 |
| `impression_generation` | Generate impression from findings | Qwen2.5-3B | 256 |
| `temporal_comparison` | Identify changes across serial reports | Qwen3-4B | 256 |
| `radiology_ner` | Extract entities, observations, changes | Qwen3-4B | 256 |
| `n_staging` | Nodal staging (N0/N1/N2) from CT reports | Qwen2.5-3B | 64 |
| `m_staging` | Metastatic staging (M0/M1) from CT reports | Qwen2.5-3B | 10 |
| `radiology_nli` | Natural language inference on report pairs | Qwen2.5-3B | 10 |
| `abnormality_detection` | Multi-label chest X-ray abnormality classification | Qwen3-4B | 256 |
| `radiology_qa` | Radiology question answering | Qwen2.5-3B | 256 |

## Key Findings

1. **LoRA fine-tuning dramatically improves performance** over zero-shot baselines across all tasks
2. **Complementary model strengths**: Qwen2.5 excels at generation, Qwen3 at extraction — a task-routed ensemble is optimal
3. **Few-shot prompting hurts** fine-tuned models (-5pp RADS accuracy), demonstrating LoRA > in-context learning for domain specialization
4. **CPU deployment is practical**: 1.8-2.4 GB models run at 4-8 tok/s on consumer hardware

## Training Details

- **Base models**: Qwen2.5-3B-Instruct, Qwen3-4B
- **Method**: LoRA (rank=64, alpha=128, all 7 projection matrices, dropout=0.05)
- **Trainable params**: 1.2-1.6% of total
- **Training data**: 161,586 samples from 12 public datasets across 9 tasks
- **Hardware**: Single NVIDIA RTX 6000 Ada (48 GB), ~26 hours per model
- **Adapter size**: ~240 MB per model

## Citation

```bibtex
@article{gupta2026radlite,
  title={RadLite: Multi-Task LoRA Fine-Tuning of Small Language Models for CPU-Deployable Radiology AI},
  author={Gupta, Pankaj and Bose, Kartik},
  journal={arXiv preprint arXiv:2605.00421},
  year={2026}
}
```

## License

This project is licensed under [CC BY-NC-SA 4.0](LICENSE). Model weights are derived from [Qwen2.5](https://huggingface.co/Qwen) and [Qwen3](https://huggingface.co/Qwen) base models, subject to their respective licenses.
