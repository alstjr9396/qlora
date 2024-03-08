import wandb
import torch
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, GenerationConfig
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer
import warnings
warnings.filterwarnings("ignore")

data = load_dataset('beomi/KoAlpaca-v1.1a')
train_data = data['train'].map(
    lambda x: {'text': f"### 질문 : {x['instruction']}\n\n### 응답 : {x['output']} <|endoftext|>"}
)

model_id = "beomi/llama-2-ko-70b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,           
    bnb_4bit_quant_type="nf4",   
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto", 
    trust_remote_code=True, 
)  

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
    

lora_alpha = 256 # scaling factor for the weight matrices
lora_dropout = 0.2 # dropout probability of the LoRA layers
lora_rank = 64 # dimension of the low-rank matrices

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_rank,
    bias="none",  # setting to 'none' for only training weight params instead of biases
    task_type="CAUSAL_LM",
    target_modules=[
      "q_proj",
      "k_proj",
      "v_proj",
      "o_proj",
      "up_proj",
      "down_proj"
    ]
)

peft_model = get_peft_model(model, peft_config)
print_trainable_parameters(peft_model)


output_dir = "outputs"
per_device_train_batch_size = 4 # reduce batch size by 2x if out-of-memory error
gradient_accumulation_steps = 8  # increase gradient accumulation steps by 2x if batch size is reduced
optim = "paged_adamw_32bit" # activates the paging for better memory management
save_strategy="steps" # checkpoint save strategy to adopt during training
save_steps = 1000 # number of updates steps before two checkpoint saves
logging_steps = 10  # number of update steps between two logs if logging_strategy="steps"
learning_rate = 1e-4  # learning rate for AdamW optimizer
max_grad_norm = 0.3 # maximum gradient norm (for gradient clipping)
max_steps = 14000        # training will happen for 320 steps
warmup_ratio = 0.05 # number of steps used for a linear warmup from 0 to learning_rate
lr_scheduler_type = "cosine"  # learning rate scheduler

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    #do_eval=True,
    #evaluation_strategy="steps",
    #eval_steps=100,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    bf16=True,
    max_grad_norm=max_grad_norm,
    num_train_epochs=1,
    #max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    report_to="wandb",
    run_name="llama-2-ko-70b-mnote-v1",
    lr_scheduler_type=lr_scheduler_type
)

trainer = SFTTrainer(
    model=peft_model,
    train_dataset=train_data,
    #eval_dataset=test_data,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_arguments,
)

# upcasting the layer norms in torch.bfloat16 for more stable training
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.bfloat16)
               
        
peft_model.config.use_cache = False
trainer.train()

peft_model.eval()
peft_model.config.use_cache = True


trainer.save_model(output_dir="qlora/llama-2-ko-70b-qlora-v1")
tokenizer.save_pretrained("qlora/llama-2-ko-70b-qlora-v1", legacy_format=False)
