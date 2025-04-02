import os
import torch
import torch.nn as nn
import wandb
from torch.utils.data import Sampler
import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional
import shutil
import deepspeed

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]

def evaluate(model, eval_dataloader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            value = batch['value'].to(device)
            if 'images' in batch:
                images = batch['images'].to(device)
            else:
                images = None
                
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                images=images,
                value=value
            )
            
            mse_loss = nn.MSELoss()(outputs.logits, value)
            
            batch_size = attention_mask.size(0)
            total_loss += mse_loss.item() * batch_size
            total_samples += batch_size
            
    avg_loss = total_loss / total_samples
    return avg_loss

class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVAValueTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_eval_loss = float('inf')
        self.best_eval_step = -1

    def compute_loss(self, model, inputs, return_outputs=False):
                
        output = model(**inputs)
        
        value = inputs.pop("value")
        
        # print(f"predicted value: {output.logits}")
        # print(f"value label: {value}")
        
        # loss_fct = nn.MSELoss()
        # loss = loss_fct(output.logits, value)
        
        # MSE损失
        mse_loss = nn.MSELoss()(output.logits, value)
        
        
        # KL散度损失,让预测分布更接近真实分布
        # kl_loss = nn.KLDivLoss(reduction='batchmean')(
        #     F.log_softmax(output.logits, dim=0),
        #     F.softmax(value, dim=0)
        # )
        
        loss = mse_loss
        
        if self.args.local_rank == 0 and self.state.global_step % self.args.logging_steps == 0:
            logs = {}
            logs["loss"] = loss.item()
            logs["mse_loss"] = mse_loss.item()
            # logs["huber_loss"] = huber_loss.item()
            # logs["kl_loss"] = kl_loss.item()
            logs["epoch"] = self.state.epoch  
            
            wandb.log(logs, step=self.state.global_step)
       
        return (loss, output) if return_outputs else loss
    
    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        print(f"loss: {loss.item()}")
        
        if self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0 and self.args.local_rank == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            wandb.log({
                    "learning_rate": current_lr
                }, step=self.state.global_step)
            
        return loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            return {}
            
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        eval_loss = evaluate(self.model, eval_dataloader, self.args.device)
        
        # if eval_loss < self.best_eval_loss:
        #     self.best_eval_loss = eval_loss
        #     self.best_eval_step = self.state.global_step
            
            # if self.args.local_rank == 0:  
            #     output_dir = os.path.join(self.args.output_dir, "val-best-model")
            #     if os.path.exists(output_dir):
            #         try:
            #             shutil.rmtree(output_dir)
            #             logger.info(f"Previous best model removed: {output_dir}")
            #         except Exception as e:
            #             logger.warning(f"Error removing previous best model: {e}")
                
            #     os.makedirs(output_dir, exist_ok=True)
                
                # if self.deepspeed:
                #     torch.cuda.synchronize()
                #     self.save_model(output_dir)
                    
                #TODO: save best_val_model
                # if self.deepspeed:
                #     with deepspeed.zero.GatheredParameters(self.model.parameters(), modifier_rank=0):
                #         state_dict = self.accelerator.get_state_dict(self.deepspeed)
                # else:
                #     state_dict = self.model.state_dict()
                # self.model.config.save_pretrained(output_dir)
                
        if torch.distributed.is_initialized():
            eval_loss_tensor = torch.tensor(eval_loss).to(self.args.device)
            torch.distributed.all_reduce(eval_loss_tensor, op=torch.distributed.ReduceOp.SUM)
            eval_loss = eval_loss_tensor.item() / torch.distributed.get_world_size()
            
        if self.args.local_rank == 0:
            os.makedirs(self.args.output_dir, exist_ok=True)
            with open(os.path.join(self.args.output_dir, "best_eval_loss.txt"), "a") as f:
                from datetime import datetime
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                f.write(f"\n[{current_time}] ")
                f.write(f"Step: {self.state.global_step}, Current eval loss: {eval_loss:.6f}\n")
                if eval_loss < self.best_eval_loss:
                    f.write(f"New best model! Previous best: {self.best_eval_loss:.6f} @ {self.best_eval_step}")
                f.write("-" * 50) 
            
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.best_eval_step = self.state.global_step
        
        metrics = {
            "eval_loss": eval_loss,
            "best_val_loss": self.best_eval_loss,
        }
        
        if self.args.local_rank == 0:
            wandb.log(metrics, step=self.state.global_step)
            
        return metrics
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVAValueTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVAValueTrainer, self)._save(output_dir, state_dict)
