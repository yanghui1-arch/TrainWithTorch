import random, json
import torch
import numpy as np
import time
import torch.nn.functional as F
from sklearn.metrics import classification_report, precision_score, f1_score, recall_score, accuracy_score
from torch.optim import AdamW
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
# from datasets import load_dataset, Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from utils import GVCDataset


setting = {
    "task_type":"articles",# "articles", "charge", "main_penalty_type", "probation", "additional_penalty_types"
    "device" : "cuda:0",
    "lr":1e-5,
    "batch_size":32,
    "max_input_length":1024,
    "model_path" : "../modelfiles/lawformer", 
    "train_data_path" : "datasets/train.json",
    "val_data_path" : "datasets/val.json",
    "test_data_path" : "datasets/test.json",
    "label_path":"datasets/article_labels.txt",
    "ckp_path":"outputs/retriever",
    "records_path":"outputs/retriever",
    "logits_th":0, 
    "iter_step":1000,
    "num_epochs" : 8
}
# print(f"train {setting['task_type']} prediction model on {setting['device']}")

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(2333)

                        
class Trainer:
    def __init__(self, device, model, train_dl, val_dl, test_dl, optimizer, iter_step) -> None:
        self.device = device
        self.model = model.to(device)
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.optimizer = optimizer
        self.iter_step = iter_step
        self.cur_step = 0
        self.cur_min_loss = float("inf")
        self.val_loss = []
        self.test_loss = []
    
    def train(self):
        self.model.train()
        num_training_steps = setting["num_epochs"] * len(self.train_dl)
        self.process_bar = tqdm(range(num_training_steps))
        self.time_stamp = time.time()
        for epoch in range(setting["num_epochs"]):
            self._run_epoch() # train
            # torch.save(self.model, f"outputs/models/ckp_{epoch}.pkl")

    def _run_epoch(self):
        for inputs, targets in self.train_dl:
            self._run_batch(inputs, targets)
            if (self.cur_step+1)%self.iter_step==0: #每1k步评估一次
                self.model.eval()
                vl = self._run_eval(self.val_dl, "val")
                self.val_loss.append(vl)
                if vl<self.cur_min_loss:
                    self._save_ckp() # save model
                    self.cur_min_loss = vl
                    self._run_eval(self.test_dl, "test")
                self.model.train()
                print(f"time consuming: {round((time.time()-self.time_stamp)/60,2)}min")
                self.time_stamp = time.time()
            self.cur_step+=1

    def _run_batch(self, inputs, targets):
        inputs = {k:v.to(self.device) for k, v in inputs.items()}
        targets = targets.to(self.device)
        self.optimizer.zero_grad()
        logits = self.model(**inputs)["logits"]
        loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        loss.backward()
        # update parameters of requires_grad = True
        self.optimizer.step()
        self.process_bar.update(1)

    def _run_eval(self, data_loader, mode):
        total_loss=0
        Y = []
        Y_hat = []
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                targets = targets.to(self.device)
                logits = self.model(**inputs)["logits"]
                loss = F.binary_cross_entropy_with_logits(logits, targets.float())
                total_loss+=loss.item()
                Y.extend(targets.tolist())
                preds = logits>setting["logits_th"]
                Y_hat.extend(preds.tolist())
        acc = f"Acc: {round(accuracy_score(y_pred=Y_hat, y_true=Y), 4)}"
        p = f"P: {round(precision_score(y_pred=Y_hat, y_true=Y, average='samples'), 4)}"
        f1 = f"F1: {round(f1_score(y_pred=Y_hat, y_true=Y, average='samples'), 4)}"
        r = f"R: {round(recall_score(y_pred=Y_hat, y_true=Y, average='samples'), 4)}"
        avg_loss = round(total_loss/len(data_loader), 4)
        records = f"step: {self.cur_step+1} | {mode}_loss: {avg_loss} | {acc} | {p} | {r} | {f1}"
        with open(setting["records_path"]+ "/" + f"{setting['task_type']}_train_records.txt", "a", encoding="utf-8") as fi:
            fi.write(records+"\n")
        print(records)
        return avg_loss
    
    def _save_ckp(self):
        # save model
        path = setting["ckp_path"] + "/" + f"{setting['task_type']}_step_{self.cur_step+1}.pkl"
        torch.save(self.model, path)

def load_trainer():
    print("loading dataset...")
    train_ds = GVCDataset(setting["task_type"], setting["train_data_path"], setting['label_path'], max_input_length=setting["max_input_length"], tk_path=setting["model_path"])
    train_dl = DataLoader(train_ds, batch_size=setting["batch_size"],shuffle=True, collate_fn=train_ds.collate_fn)
    val_ds = GVCDataset(setting["task_type"], setting["val_data_path"], setting['label_path'], max_input_length=setting["max_input_length"], tk_path=setting["model_path"])
    val_dl = DataLoader(val_ds, batch_size=2*setting["batch_size"],shuffle=False, collate_fn=train_ds.collate_fn)
    test_ds = GVCDataset(setting["task_type"], setting["test_data_path"], setting['label_path'], max_input_length=setting["max_input_length"], tk_path=setting["model_path"])
    test_dl = DataLoader(test_ds, batch_size=2*setting["batch_size"],shuffle=False, collate_fn=train_ds.collate_fn)
    model = AutoModelForSequenceClassification.from_pretrained(setting["model_path"],
                                                               num_labels=len(train_ds.label_text))
    optimizer = AdamW(model.parameters(), lr=setting["lr"])
    
    trainer = Trainer(device=setting["device"],
                      model=model,
                      train_dl=train_dl,
                      val_dl=val_dl,
                      test_dl=test_dl,
                      optimizer=optimizer,
                      iter_step=setting["iter_step"])
    return trainer



if __name__=="__main__":
    trainer = load_trainer()
    trainer.train()
    # for th in [-1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.5]:
    #     get_predictions("articles", th)
        
    
    


