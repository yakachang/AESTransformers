# from typing import Any, Dict, List, Optional, Tuple, Union

# import torch
from torch import nn

# from torch.utils.data import DistributedSampler, RandomSampler

from transformers import logging

# from transformers.integrations import is_fairscale_available
# from transformers.models.fsmt.configuration_fsmt import FSMTConfig
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

# from transformers.trainer_pt_utils import get_tpu_sampler
# from transformers.training_args import ParallelMode
# from transformers.utils import is_torch_tpu_available

from transformers import Seq2SeqTrainer


logger = logging.get_logger(__name__)

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
}


class CustomTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # if self.label_smoother is not None and "labels" in inputs:
        if "labels" in inputs:
            labels = inputs["labels"]
        else:
            labels = None

        # outputs = model(**inputs)
        # # Save past state if it exists
        # # TODO: this needs to be fixed and made cleaner later.
        # if self.args.past_index >= 0:
        #     self._past = outputs[self.args.past_index]

        # if labels is not None:
        #     print(f"labels: {labels}")
        #     print(f"outputs: {outputs[0]}")
        #     loss = self.label_smoother(outputs, labels)
        # else:
        #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #     loss = outputrs["loss"] if isinstance(outputs, dict) else outputs[0]
        logits = model(**inputs, use_cache=False)[0]
        print(f"logits: {logits}")
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
        loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return (loss, logits) if return_outputs else loss
