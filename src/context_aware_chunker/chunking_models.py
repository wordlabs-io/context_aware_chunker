import abc 
from abc import ABC, abstractmethod
 
import logging  

from transformers import AutoTokenizer, T5ForConditionalGeneration, BertLMHeadModel

import torch

class ChunkerModel(ABC): 
 
    def get_model_details(self): 
        return {
            'model_name': self.model_name,
            'cuda_available': self.check_cuda()
        }
    
    def get_device(self):
        return self.device

    def check_cuda(self):
        return torch.cuda.is_available()

    def get_logger(self):
        self.logger = logging.getLogger()

    def calculate_perplexity(self, logits, targets, vocab_size):
           
        softmaxed = self.softmaxer(logits)
        vocab_size = softmaxed.size()[-1]
        loss_val = torch.sum(torch.log(softmaxed) * -1 * torch.nn.functional.one_hot(targets, vocab_size))
        perplexity = torch.exp(loss_val / targets.numel())
        return perplexity.item()

    def chunk(self, split_text: [str]):

        cur_segment = None
        op_segments = []

        for idx, sentence in enumerate(split_text):
            if cur_segment is None:
                cur_segment = sentence

            text1 = cur_segment
            text2 = split_text[idx + 1]

            ppl_ip, ppl_combined = self.find_ppl_for_splits(text1, text2)

            if ppl_ip >= ppl_combined:
                cur_segment = cur_segment + text2
            else:
                op_segments.append(cur_segment)
                cur_segment = None

            if idx == len(split_text) - 2:
                if cur_segment is None:
                    op_segments.append(text2)
                break
        return op_segments

    @abstractmethod
    def load_model(self):
        pass

    def find_ppl_for_splits(self, text1: str, text2: str):
                
        text3 = text1 + text2
        inputs = self.tokenizer([text1, text2, text3], return_tensors="pt", padding=True, truncation = True)
        
        inputs.to(self.device)
        
        outputs = self.model(**inputs, labels=inputs["input_ids"])

        op_independent = outputs.logits[0:2,:, :]
        op_joined = outputs.logits[2,:, :]

        ip_ppl = self.calculate_perplexity(op_independent, inputs["input_ids"][0:2], self.vocab_size)
        combined_ppl = self.calculate_perplexity(op_joined, inputs["input_ids"][2], self.vocab_size)

        return ip_ppl, combined_ppl
    
class T5ChunkerModel(ChunkerModel): 

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_model_loaded = False
        self.get_logger()
        self.load_model()
        self.softmaxer = torch.nn.Softmax()
        self.vocab_size = 32128

    def load_model(self):
        
        if self.is_model_loaded is True:
            self.logger.warning("WARNING: The model is already loaded")
            return

        if torch.cuda.is_available(): 
            self.device = "cuda:0" 
            torch.cuda.empty_cache()
        else: 
            self.device = "cpu"
            self.logger.warning("WARNING: Unable to find CUDA, defaulting to CPU. This might be significantly slower") 
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.is_model_loaded = True

class BertChunkerModel(ChunkerModel): 

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_model_loaded = False
        self.get_logger()
        self.load_model()
        self.softmaxer = torch.nn.Softmax()
        self.vocab_size = 30522

    def load_model(self):
        
        if self.is_model_loaded is True:
            self.logger.warning("WARNING: The model is already loaded")
            return

        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()
            self.device = "cuda:0" 
        else: 
            self.device = "cpu"
            self.logger.warning("WARNING: Unable to find CUDA, defaulting to CPU. This might be significantly slower") 
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = BertLMHeadModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.is_model_loaded = True

