import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class GVCDataset(Dataset):
    def __init__(self,label_type, data_path, label_path, max_input_length, tk_path):
        super().__init__()
        assert label_type in ["articles", "charge", "main_penalty_type", "probation", "additional_penalty_types"]
        self.label_type = label_type
        self.data_path = data_path
        self.label_path = label_path
        self.max_input_length = max_input_length
        self.tokenizer = AutoTokenizer.from_pretrained(tk_path) # setting["model_path"]
        self.inputs, self.label_indices = self.load_data()

    def __getitem__(self, index):
        return self.inputs[index], self.label_indices[index]
    
    def __len__(self):
        return len(self.label_indices)
    
    def collate_fn(self, batch):
        facts = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        tokenized_facts = self.tokenizer(facts, max_length=self.max_input_length, padding="max_length", truncation=True, return_tensors="pt")
        label_indices = torch.tensor(labels,dtype=torch.float32)
        return tokenized_facts, label_indices
        
    def load_data(self):
        inputs = []
        label_indice= []
        # 加载标签
        with open(self.label_path, "r", encoding="utf-8") as fi:
            self.label_text = json.loads(fi.read())
            print(f"{len(self.label_text)} articles")
        # 加载数据
        with open(self.data_path,"r",encoding="utf-8") as fi:
            for line in fi.readlines():
                case = json.loads(line)
                fact = case["bg_info"]+case["fact"]
                inputs.append(fact)
                if self.label_type in ["additional_penalty_types","articles"]:
                    temp = [0]*len(self.label_text)
                    for item in case["annotation"]:
                        temp[self.label_text.index(item[2][0])] = 1
                    label_indice.append(temp)
                if self.label_type in ["charge", "probation", "main_penalty_type"]:# , ,
                    temp = [0]*len(self.label_text)
                    temp[self.label_text.index(case["key_points"][self.label_type])] = 1
                    label_indice.append(temp)
                if self.label_type == "multi_task": # 多任务类型
                    pass
        return inputs, label_indice