import logging

import torch
import torch.distributed as dist

from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.utils import print_on_rank0

logger = logging.getLogger(__name__)


class BF16Optimizer:
    """AdamW over fp32 master copies of the bf16 trainable params, with grad
    clipping and cosine warmup scheduling."""

    def __init__(
        self,
        model=None,
        lr=None,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
        param_groups=None,
    ):
        # TODO: For now, we only support cosine annealing warmup lr scheduler and AdamW optimizer
        # TODO: We should make these parameters configurable
        #   These magic numbers: weight_decay=0.0, max_grad_norm=0.5, total_steps=800k, warmup_steps=12k are copied from
        #   https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/ds_config.json
        if param_groups is not None:
            self.model = None
            self.model_params = []
            fp32_param_groups = []
            for group in param_groups:
                params = [p for p in group["params"] if p.requires_grad]
                self.model_params.extend(params)
                fp32_params = [p.detach().clone().to(torch.float32) for p in params]
                for mp in fp32_params:
                    mp.requires_grad = True
                fp32_group = {k: v for k, v in group.items() if k != "params"}
                fp32_group["params"] = fp32_params
                fp32_param_groups.append(fp32_group)
            self.optimizer = torch.optim.AdamW(fp32_param_groups)
        else:
            if model is None or lr is None:
                raise ValueError(
                    "BF16Optimizer requires either a model and lr, or param_groups"
                )
            self.model = model
            self.model_params = [p for p in model.parameters() if p.requires_grad]
            self.fp32_params = [
                p.detach().clone().to(torch.float32) for p in self.model_params
            ]
            for mp in self.fp32_params:
                mp.requires_grad = True
            self.optimizer = torch.optim.AdamW(
                self.fp32_params, lr=lr, weight_decay=weight_decay
            )
            fp32_param_groups = self.optimizer.param_groups
        self.max_grad_norm = max_grad_norm
        self.fp32_params = [p for group in fp32_param_groups for p in group["params"]]
        self.last_grad_norm = None
        self._grad_norm_process_group = None
        self._reduce_grad_norm_across_ranks = True
        self.scheduler = CosineAnnealingWarmupLR(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
        )

    def configure_grad_norm_reduction(
        self, *, process_group=None, enabled: bool = True
    ) -> None:
        """Configure the group that owns disjoint gradient shards.

        FSDP backends disable the reduction for replicated/NO_SHARD parameters.
        """
        self._grad_norm_process_group = process_group
        self._reduce_grad_norm_across_ranks = enabled

    def _clip_grad_norm(self):
        """Clip all FSDP shards with one global L2-norm coefficient."""
        grads = [mp.grad for mp in self.fp32_params if mp.grad is not None]
        if grads:
            total_norm_sq = torch.stack([grad.square().sum() for grad in grads]).sum()
        else:
            device = self.fp32_params[0].device if self.fp32_params else "cpu"
            total_norm_sq = torch.zeros((), dtype=torch.float32, device=device)

        if (
            self._reduce_grad_norm_across_ranks
            and dist.is_available()
            and dist.is_initialized()
        ):
            dist.all_reduce(
                total_norm_sq,
                op=dist.ReduceOp.SUM,
                group=self._grad_norm_process_group,
            )

        total_norm = total_norm_sq.sqrt()
        clip_coef = torch.clamp(self.max_grad_norm / (total_norm + 1e-6), max=1.0)
        for grad in grads:
            grad.mul_(clip_coef)
        return total_norm

    def step(self):
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                mp.grad = (
                    p.grad.detach().to(torch.float32) if p.grad is not None else None
                )
        grad_norm = self._clip_grad_norm()
        self.last_grad_norm = grad_norm.detach()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.data.copy_(mp.data.to(p.dtype))
                p.grad = None
        return self.last_grad_norm

    def load_state_dict(self, state_dict):
        """Restore optimizer/scheduler state and, when present, the rank-local
        fp32 master params; without them the masters are re-cloned from the
        bf16 weights and the resume is not numerically faithful."""
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        print_on_rank0("Successfully loaded optimizer state_dict.")
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        print_on_rank0("Successfully loaded scheduler state_dict.")
        saved_fp32 = state_dict.get("fp32_params")
        if saved_fp32 is not None:
            if len(saved_fp32) != len(self.fp32_params):
                raise ValueError(
                    f"checkpoint carries {len(saved_fp32)} fp32 master params "
                    f"but this rank has {len(self.fp32_params)}"
                )
            with torch.no_grad():
                for i, (saved, mp) in enumerate(zip(saved_fp32, self.fp32_params)):
                    if saved.shape != mp.shape:
                        raise ValueError(
                            f"fp32 master param {i} shape mismatch: checkpoint "
                            f"{tuple(saved.shape)} vs current {tuple(mp.shape)}"
                        )
                    mp.data.copy_(saved.to(mp.device, mp.dtype))
        else:
            logger.warning(
                "checkpoint has no fp32_params; re-cloning master params from "
                "bf16 weights — resume will not be numerically faithful"
            )
            with torch.no_grad():
                for p, mp in zip(self.model_params, self.fp32_params):
                    mp.data.copy_(p.detach().to(torch.float32))

    def state_dict(self):
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            # rank-local fp32 masters; without them a resume re-quantizes from bf16
            "fp32_params": [t.detach().cpu() for t in self.fp32_params],
        }

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]
