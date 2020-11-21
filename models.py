import torch
try:
    import apex
    from apex.contrib.sparsity import ASP
except Exception as e:
    print(e, "Warning: failed to import Apex (https://github.com/NVIDIA/apex), fused optimizers not available.")
import pytorch_lightning as pl
import transformers

class TransformerModel(pl.LightningModule):
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2, fused_opt=True, sparse=False):
        super().__init__()
        self.transformer = transformers.AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                                           num_labels=num_labels,
                                                                                           output_attentions=False,
                                                                                           return_dict=True)
        self.fused_opt = fused_opt
        self.sparse = sparse
        if self.fused_opt:
            self.opt = apex.optimizers.FusedAdam(self.parameters(), lr=1e-5, adam_w_mode=True)
        else:
            self.opt = torch.optim.AdamW(self.parameters(), lr=1e-5)
        if self.sparse:
            print("Experimental: automatic sparsity enabled!")
            print("For more info, see: https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity")
            self.transformer = self.transformer.cuda()
            ASP.prune_trained_model(self.transformer, self.opt)

    def forward(self, input_ids, attention_mask, token_type_ids):
        logits = self.transformer(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)["logits"]
        return logits
    
    def configure_optimizers(self):
        opt = self.opt
        return opt
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["label"]
        attention_mask = batch["attention_mask"]
        outputs = self.transformer(input_ids, attention_mask=attention_mask, labels=labels)
        loss_value = torch.nn.functional.cross_entropy(outputs["logits"], labels)
        return loss_value
    
    def training_epoch_end(self, outputs):
        loss_list = [o["loss"] for o in outputs]
        final_loss = sum(loss_list)/len(outputs)
        self.log("train_loss", final_loss)
    
    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["label"]
        attention_mask = batch["attention_mask"]
        outputs = self.transformer(input_ids, attention_mask=attention_mask, labels=labels)
        loss_value = torch.nn.functional.cross_entropy(outputs["logits"], labels)
        return loss_value
    
    def test_epoch_end(self, outputs):
        final_loss = sum(outputs)/len(outputs)
        self.log("test_loss", final_loss)
    