import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import logging

class LoRAFTModel(object):

    def __init__(self, pretrained_model_name_or_path, lora_config: LoraConfig,
                 quantization_config: BitsAndBytesConfig = None):

        # Load the base model with quantization if provided
        if quantization_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                quantization_config=quantization_config,
                device_map='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                device_map='cuda' if torch.cuda.is_available() else 'cpu'
            )
        self.model = prepare_model_for_kbit_training(self.model)

        #initialize LoRA
        self.model = get_peft_model(self.model, lora_config)
        trainable_params_number, all_params_number = self.model.get_nb_trainable_parameters()
        logging.info(
            f"LoRA model initialized. Trainable parameters: {trainable_params_number}, All parameters: "
            f"{all_params_number}, trainable ratio: {trainable_params_number / all_params_number:.2%}")

    def forward(self, input_ids):
        pass

    def generate(self, input_ids):
        pass
