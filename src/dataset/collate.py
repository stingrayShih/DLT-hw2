"""
# Author: Yinghao Li
# Modified: September 30th, 2023
# ---------------------------------------
# Description: collate function for batch processing
"""

import torch
from transformers import DataCollatorForTokenClassification

from .batch import unpack_instances, Batch


class DataCollator(DataCollatorForTokenClassification):
    def __call__(self, instance_list: list[dict]):
        tk_ids, attn_masks, lbs = unpack_instances(instance_list, ["bert_tk_ids", "bert_attn_masks", "bert_lbs"])

        # Update `tk_ids`, `attn_masks`, and `lbs` to match the maximum length of the batch.
        # The updated type of the three variables should be `torch.int64``.
        # Hint: some functions and variables you may want to use: `self.tokenizer.pad()`, `self.label_pad_token_id`.
        # --- TODO: start of your code ---
        
        batch = self.tokenizer.pad(
            {"input_ids": tk_ids, "attention_mask": attn_masks, 'labels':lbs},
            padding=True
        )

        tk_ids = torch.LongTensor(batch['input_ids'])
        #print("tk_ids",tk_ids.shape, tk_ids.dtype)
        
        attn_masks = torch.LongTensor(batch['attention_mask'])
        #print("attn_masks",attn_masks.shape, attn_masks.dtype)

        max_len=tk_ids.shape[1]
        #print('max_len',max_len)
        for i in range(len(lbs)):
          lbs[i] = lbs[i]+[self.label_pad_token_id]*(max_len-len(lbs[i]))
          #print(lbs[i])

        lbs=torch.LongTensor(lbs)
        #print("lbs", lbs.shape, lbs.dtype)

        # --- TODO: end of your code ---

        return Batch(input_ids=tk_ids, attention_mask=attn_masks, labels=lbs)
