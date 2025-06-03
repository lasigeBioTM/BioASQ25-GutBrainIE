#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/05/01 15:23:02
@author: SIRConceicao
'''
import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

#===========================================================================================
# ** QWEN **
#===========================================================================================
class Qwen_Pipeline:
    def __init__(self, model_name,configs):
        self.configs= configs
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def inference(self, messages):
        text = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.configs["max_new_tokens"],
            temperature=self.configs['temperature'],
            # TopP=0.95,
            # TopK=20,
            # MinP=0
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return thinking_content, content

#===========================================================================================
# ** Mistral **
#===========================================================================================
class Mistral_Pipeline:
    def __init__(self, model_name,configs):
        load_dotenv()
        HUGGINGFACE_TOKEN=os.getenv('HUGGINGFACE_TOKEN')
        login(token=HUGGINGFACE_TOKEN)

        self.configs= configs

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.configs["quantization"]:
            print('Quantized Model')
            bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            )

            self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                            device_map="auto",
                                                            quantization_config=bnb_config)
        else:
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
        
    def inference(self, message):
        input_text = self.tokenizer.apply_chat_template(message, tokenize=False)
        model_inputs = self.tokenizer(input_text, return_tensors="pt").to("cuda")

        attention_mask = model_inputs["attention_mask"]
        generated_ids = self.model.generate(model_inputs["input_ids"],
                                    max_new_tokens=self.configs["max_new_tokens"],
                                    do_sample=True,
                                    temperature=self.configs['temperature'],
                                    attention_mask=attention_mask,
                                    pad_token_id=self.tokenizer.eos_token_id
                                    )
        
        output=self.tokenizer.batch_decode(generated_ids)[0]

        return output
    
#===========================================================================================
# ** LLAMA **
#===========================================================================================
class Llama_Pipeline:
    def __init__(self, model_name,configs):
        load_dotenv()
        HUGGINGFACE_TOKEN=os.getenv('HUGGINGFACE_TOKEN')
        #login(token=HUGGINGFACE_TOKEN)
  
        self.configs= configs

        self.tokenizer = AutoTokenizer.from_pretrained(model_name,token=HUGGINGFACE_TOKEN)
        
        if self.configs["quantization"]:
            print('Quantized Model')
            bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
            )

            self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                            device_map="cuda",
                                                            quantization_config=bnb_config,
                                                            token=HUGGINGFACE_TOKEN)

        else:
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="cuda",
            )
        print(next(self.model.parameters()).device)

    def inference(self,message):
        terminators = [self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        input_ids = self.tokenizer.apply_chat_template(
                conversation=message, 
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
                ).to(self.model.device)
        
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.configs["max_new_tokens"],
            eos_token_id=terminators,
            do_sample=True,
            temperature=self.configs['temperature'],
            top_p=0.9,
            )
        
        generated_output = outputs[0][input_ids.shape[-1]:]
        output=self.tokenizer.decode(generated_output, skip_special_tokens=True)

        return output

#===========================================================================================
# ** OTHER **
#===========================================================================================