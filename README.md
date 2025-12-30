# Qwen2.5-Coder LoRA Fine-Tuning Project 

This repository contains the implementation of a LoRA (Low-Rank Adaptation) fine-tuning project for the **Qwen2.5-Coder-1.5B-Instruct** model. The goal of this project is to enhance the code generation capabilities of the base model using two different dataset strategies: **Deep Instruction** (Chain-of-Thought) and **Diverse Instruction**.

This project was conducted as part of the **CEN471 - Natural Language Processing** course.

## 🛠️ Environment & Hardware

- **Platform:** Google Colab Pro
- **GPU:** NVIDIA A100 (80GB)
- **Training Method:** LoRA (PEFT)
- **Precision:** `bfloat16` (Brain Floating Point)

## 📂 Repository Structure

### 🔹 Code & Configuration
- `train.py`: The main training script. (Used for both Deep and Diverse datasets by modifying the configuration).
- `requirements.txt`: Dependencies required to run the project.

*(Note: The evaluation logic uses the `livecodebench_eval.py` script from the [original CodeGen repository](https://github.com/naholav/CodeGen).)*

### 🔹 Data & Logs (Evidence)
- `deep_training_log.csv`: Raw training logs (Loss values) for Deep Instruction model.
- `diverse_training_log.csv`: Raw training logs (Loss values) for Diverse Instruction model.
- `final_results/`: **Full evaluation outputs.** This directory contains:
    - Generated Python codes for all 41 problems.
    - Detailed JSONL logs for every test case.
    - Summary JSON files with Pass@1 scores.

### 🔹 Visual Analysis
- `deeplossfinal.png`: Loss analysis graph for the Deep Instruction model.
- `diverselossfinal.png`: Loss analysis graph for the Diverse Instruction model.

## ⚠️ Implementation Details & Modifications

To adapt the project to the Colab environment and specific hardware constraints, I made the following modifications:

1.  **Google Drive Integration:**
    I configured the training scripts to save checkpoints directly to Google Drive (`/content/drive/MyDrive/...`) to ensure data persistence during long training sessions.

2.  **Flash Attention Adjustment:**
    Although trained on an A100 GPU, I encountered library compatibility issues with Flash Attention 2 during the inference/evaluation phase. Therefore, I used a modified version of the evaluation script where the `use_flash_attention_2=True` argument was removed. The evaluation runs in standard `bfloat16` mode.

## 📊 Benchmark Results (Pass@1)

I evaluated the models using the **LiveCodeBench (AtCoder - Easy)** dataset consisting of 41 coding problems. Detailed evaluation data is available in the `final_results` directory.

| Model | Best Checkpoint | Pass@1 Score | Problems Solved |
| :--- | :--- | :--- | :--- |
| **Base Model (Qwen2.5-Coder)** | - | **26.83%** | 11 / 41 |
| **Deep Instruction (My Model)** | **Step-200** | **34.15%** 🏆 | **14 / 41** |
| **Diverse Instruction** | Step-200 | **29.27%** | 12 / 41 |

**Conclusion:**
The **Deep Instruction** model significantly outperformed both the Base model (+7.3%) and the Diverse Instruction model. This demonstrates that training with Chain-of-Thought (CoT) data is more effective for reasoning-based coding tasks than simply increasing data diversity.

## 📈 Training Analysis & Overfitting

I monitored Train, Validation, and Test losses throughout the training process. The raw data for these graphs is available in the `.csv` log files.

### 1. Deep Instruction Model
![Deep Instruction Loss Graph](deeplossfinal.png)

### 2. Diverse Instruction Model
![Diverse Instruction Loss Graph](diverselossfinal.png)

### 🔍 Analysis
In both training sessions, the **Test Loss (Green Line)** reached its minimum around **Step 200**. As training continued to Step 300, the Test Loss began to plateau or increase, while the benchmark performance dropped (e.g., Deep model dropped from 34.1% to 24.4%). This divergence between Training Loss (which kept decreasing) and Test Loss clearly indicates that **overfitting** started occurring after Step 200. Therefore, I selected **Step 200** as the optimal checkpoint.
### Qualitative Analysis
### 🔍 Key Findings
Reasoning vs. Robustness: The Deep model thrives on logic-heavy tasks due to its Chain-of-Thought training, while the Diverse model is more robust in following specific output formats and simple if-else patterns.

Overthinking Trap: In simple tasks (like abc370_a), the Deep model occasionally failed by "hallucinating complexity," whereas the Diverse model provided a direct and correct solution.

Knowledge Preservation: The high number of "Common Success" problems indicates that LoRA fine-tuning effectively specialized the models without causing catastrophic forgetting of basic programming concepts.
### 📊 Comparative Performance Analysis (Step-200 Checkpoints)

| Category | Count | Example Problems | Analysis & Behavioral Insights |
| :--- | :---: | :--- | :--- |
| **🤝 Common Success** | **8** | `abc365_a`, `abc372_a` | Both models successfully preserved basic Python syntax and core logic. |
| **🧠 Deep Exclusive** | **6** | `abc369_b`, `abc377_a` | Solved tasks requiring **multi-step reasoning** and **state tracking**. |
| **🎨 Diverse Exclusive** | **4** | `abc367_b`, `abc370_a` | Excelled at **pattern matching** and **strict output formatting**. |
| **🚫 Unsolved** | **23** | `abc365_b`, `abc386_b` | Problems beyond the reasoning capacity of a 1.5B scale model. |
| **🏆 Final Pass@1** | **41** | **Deep: 34.15%** | **Diverse: 29.27%** |

-	## 🔗 Trained Models (Hugging Face)

I have released the LoRA fine-tuned models on Hugging Face. You can access the checkpoints and model cards below:

- 🧠 **Deep Instruction Model (Best Performance):** [Qwen2.5-Coder-Deep-Instruct-LoRA](https://huggingface.co/deneme0001/Qwen2.5-Coder-Deep-Instruct-LoRA)  
  *(Trained on CodeGen-Deep-5K, achieved 34.15% Pass@1)*

- 🎨 **Diverse Instruction Model:** [Qwen2.5-Coder-Diverse-Instruct-LoRA](https://huggingface.co/deneme0001/Qwen2.5-Coder-Diverse-Instruct-LoRA)  
  *(Trained on CodeGen-Diverse-5K, focused on syntax robustness)*

