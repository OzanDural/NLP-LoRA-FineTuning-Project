# --- 1. KURULUMLAR ---
import torch
import os
import shutil
import glob
import math
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model

# --- 2. AYARLAR ---
BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATASET_ID    = "Naholav/CodeGen-Diverse-5K"
OUTPUT_DIR    = "/content/drive/MyDrive/CodeGen_Project/diverse_instruction/checkpoints"

SYSTEM_PROMPT = "You are an expert Python programmer. Please read the problem carefully before writing any Python code."

MAX_SEQ_LENGTH = 1024
NUM_EPOCHS = 3
BATCH_SIZE = 16
GRAD_ACCUMULATION = 1
LEARNING_RATE = 1e-4
LORA_R = 32
LORA_ALPHA = 64
LOGGING_STEPS = 20

# --- 3. MODEL VE TOKENIZER ---
print(" Model ve Tokenizer Yükleniyor ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"



model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=False
)
model.gradient_checkpointing_enable()

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- 4. DATASET HAZIRLIĞI ---
print(" Dataset Filtreleniyor...")
dataset = load_dataset(DATASET_ID)

train_ds = dataset["train"].filter(lambda x: x["split"] == "train")
valid_ds = dataset["train"].filter(lambda x: x["split"] == "valid")
test_ds  = dataset["train"].filter(lambda x: x["split"] == "test")

def preprocess_function(examples):
    inputs = examples["input"]
    solutions = examples["solution"]

    model_inputs = []
    attention_masks = []
    labels = []

    for inp, sol in zip(inputs, solutions):
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": inp}
        ]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        full_text = prompt_text + sol + tokenizer.eos_token

        tokenized_full = tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH
        )

        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]
        label = list(input_ids)

        # Maskeleme
        tokenized_prompt = tokenizer(
            prompt_text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            add_special_tokens=False
        )
        prompt_len = len(tokenized_prompt["input_ids"])

        for i in range(prompt_len):
            if i < len(label):
                label[i] = -100

        for i in range(len(label)):
            if input_ids[i] == tokenizer.pad_token_id:
                label[i] = -100

        model_inputs.append(input_ids)
        attention_masks.append(attention_mask)
        labels.append(label)

    return {
        "input_ids": model_inputs,
        "attention_mask": attention_masks,
        "labels": labels
    }

print(" Veriler İşleniyor ...")
train_tok = train_ds.map(preprocess_function, batched=True)
valid_tok = valid_ds.map(preprocess_function, batched=True)
test_tok  = test_ds.map(preprocess_function, batched=True)

# --- 5. CALLBACKS ---
class PDFCompliantLogger(TrainerCallback):
    def __init__(self, output_file):
        self.output_file = output_file
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(self.output_file, "w") as f:
            f.write("Step,Train_Loss,Valid_Loss,Test_Loss\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.output_file, "a") as f:
                step = state.global_step
                t_loss = logs.get("loss", "")
                v_loss = logs.get("eval_valid_loss", "")
                test_loss = logs.get("eval_test_loss", "")

                if t_loss != "" or v_loss != "" or test_loss != "":
                    f.write(f"{step},{t_loss},{v_loss},{test_loss}\n")

class CustomEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3):
        self.patience = patience
        self.best_loss = float('inf')
        self.early_stop_counter = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_loss = metrics.get("eval_test_loss")
        if current_loss is None: return

        print(f"\n[EarlyStopping] Best: {self.best_loss:.4f} | Current (Test): {current_loss:.4f}")

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.early_stop_counter = 0
            print(" İyileşme var! Sayaç sıfırlandı.")
        else:
            self.early_stop_counter += 1
            print(f" Loss düşmedi! ({self.early_stop_counter}/{self.patience})")
            if self.early_stop_counter >= self.patience:
                control.should_training_stop = True
                print(" EĞİTİM DURDURULUYOR (Overfitting Başladı).")

# --- 6. TRAINING ARGÜMANLARI ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,


    bf16=True,
    tf32=True,
    optim="adamw_torch",

    logging_steps=LOGGING_STEPS,
    eval_strategy="steps",
    eval_steps=LOGGING_STEPS,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=None,
    load_best_model_at_end=False,
    report_to="tensorboard"
)

# --- 7. BAŞLAT ---
if os.path.exists(f"{OUTPUT_DIR}/loss_logs.csv"):
    try: os.remove(f"{OUTPUT_DIR}/loss_logs.csv")
    except: pass

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset={"valid": valid_tok, "test": test_tok},
    callbacks=[
        PDFCompliantLogger(f"{OUTPUT_DIR}/loss_logs.csv"),
        CustomEarlyStoppingCallback(patience=3)
    ]
)

print(" Eğitim Başlıyor...")
trainer.train()

# --- 8. ÇIKIŞ VE KAYIT İŞLEMLERİ ---
print("\n Checkpoint isimleri düzenleniyor...")
checkpoints = glob.glob(f"{OUTPUT_DIR}/checkpoint-*")
steps_per_epoch = len(train_tok) // (BATCH_SIZE * GRAD_ACCUMULATION)

for cp_path in checkpoints:
    folder_name = os.path.basename(cp_path)
    if "step" in folder_name: continue

    try:
        step_num = int(folder_name.split("-")[-1])
        epoch_num = math.ceil(step_num / steps_per_epoch)

        new_name = f"checkpoint-step-{step_num}-epoch-{epoch_num}"
        new_path = os.path.join(OUTPUT_DIR, new_name)

        os.rename(cp_path, new_path)
        print(f" İsim Değiştirildi: {folder_name} -> {new_name}")
    except Exception as e:
        pass

final_path = f"{OUTPUT_DIR}/final_model_deep_a100"
trainer.model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
print(f" Eğitim Tamamlandı! Loglar: {OUTPUT_DIR}/loss_logs.csv")