import itertools
import math
import os
import socket
from copy import deepcopy
from typing import Callable, Dict, List, Tuple

import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

import ray
import torch
from transformers.trainer import get_scheduler

from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.models import Actor
# from openrlhf.trainer import PPOTrainer
from openrlhf.trainer import ReinforceTrainer
from openrlhf.trainer.ppo_utils import Experience, RemoteExperienceMakerReinforce as RemoteExperienceMaker
from openrlhf.utils import DeepspeedStrategy, blending_datasets, get_tokenizer
from openrlhf.utils.deepspeed_utils import _z3_params_to_fetch
from openrlhf.utils.distributed_util import init_process_group
from openrlhf.trainer.ppo_utils.global_envs import RUNTIME_ENV

from .launcher import BasePPORole
from .utils import get_cosine_schedule_with_warmup


class ActorReinforceTrainer(ReinforceTrainer):
    def __init__(
        self,
        *args,
        vllm_engines: List = None,
        **kwargs,
    ):
        """PPOTrainer for ray.

        Args:
            vllm_engines (List, optional): vllm engines for text generation, if not specified, generate text by actor model directly. Defaults to None.
            critic_train_remote (bool, optional): whether this actor should triger corresponding critic model training. Defaults to False.
        """
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines

        self.experience_maker = RemoteExperienceMaker(
            self.actor,
            None,
            self.reward_model,
            self.remote_rm_url,
            self.initial_model,
            self.tokenizer,
            self.prompt_max_len,
            self.kl_ctl,
            self.strategy,
            self.reward_fn,
            self.tokenizer_reward,
            vllm_engines=self.vllm_engines,
        )

        # vLLM is colocated on this GPU: weights go over CUDA IPC (_broadcast_to_vllm), no NCCL group
        torch.distributed.barrier()

    def reinforce_train(self, global_step):
        # 1. ensure all experience makers done
        self.experience_maker.flush()
        torch.distributed.barrier()

        # 2. triger remote critic model training
        # if self.critic_train_remote:
            # critic_status_ref = self.critic.fit.remote()

        # 3. actor model training (the colocated vLLM engine is asleep since the rollout: see sample_responses_bymcts)
        status = super().reinforce_train(global_step)

        # 4. broadcast weights to vllm engines
        if self.vllm_engines is not None:
            # torch.distributed.barrier()
            self._broadcast_to_vllm()

        # 5. wait remote critic model training done
        # if self.critic_train_remote:
            # status.update(ray.get(critic_status_ref))
        torch.distributed.barrier()

        return status

    def training_step(self, experience: Experience) -> Dict[str, float]:
        return self.training_step_actor(experience)

    def _broadcast_to_vllm(self):
        """Hand the new weights to the colocated, sleeping vLLM engine as CUDA IPC handles (no copy).
        Replaces the NCCL broadcast, which cannot run with both processes on one GPU."""
        from torch.multiprocessing.reductions import reduce_tensor

        assert self.strategy.args.zero_stage != 3, "IPC weight sync needs full parameters on the GPU (ZeRO-1/2)"
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        handles = [(name, reduce_tensor(p.data)) for name, p in self.actor.model.module.named_parameters()]
        for engine in self.vllm_engines:
            ray.get(engine.wake_up.remote())
            ray.get(engine.update_weights_cuda_ipc.remote(handles))
        del handles

    # def _broadcast_to_reference_model(self):
    #     model = self.actor.model.module

def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


@ray.remote(num_gpus=1, runtime_env=RUNTIME_ENV)
class ActorModelRayActor(BasePPORole):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain):
        self._setup_distributed(strategy)
        actor = Actor(
            pretrain,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            ds_config=strategy.get_ds_train_config(is_actor=True),
        )

        # configure tokenizer
        self.tokenizer_reward = get_tokenizer(strategy.args.reward_pretrain.split(",")[0], actor.model, "left", strategy)

        self.tokenizer = get_tokenizer(pretrain, actor.model, "left", strategy)

        strategy.print(actor)
        self.prepare_datasets()

        args = strategy.args

        if args.enable_ema:
            ema_model = deepcopy(actor)
        else:
            ema_model = None

        # configure optimizer
        actor_optim = strategy.create_optimizer(
            actor, lr=args.actor_learning_rate, betas=(0.9, 0.95), weight_decay=args.l2
        )

        # configure scheduler
        num_update_steps_per_episodes = len(self.prompts_dataloader) * args.max_epochs // strategy.accumulated_gradient
        max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)
        self.max_steps = max_steps

        min_actor_learning_rate_lr = getattr(args, "min_actor_learning_rate_lr", 0.1)
        
        if args.lr_scheduler_type == "cosine":
            actor_scheduler = get_cosine_schedule_with_warmup(actor_optim, num_warmup_steps=math.ceil(max_steps * 0.05), num_training_steps=max_steps, min_lr=min_actor_learning_rate_lr)
        else:
            actor_scheduler = get_scheduler(
                args.lr_scheduler_type,
                actor_optim,
                num_warmup_steps=math.ceil(max_steps * 0.05),
                num_training_steps=max_steps,
            )

        # actor_scheduler = get_scheduler(
        #     "cosine",
        #     actor_optim,
        #     num_warmup_steps=math.ceil(max_steps * 0.03),
        #     num_training_steps=max_steps,
        # )

        if args.gradient_checkpointing:
            actor.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # prepare models/optimizers...
        self.actor, self.actor_optim, self.actor_scheduler = strategy.prepare(
            (actor, actor_optim, actor_scheduler),
            is_rlhf=True,
        )

        if ema_model:
            ema_model._offload = True
            self.ema_model = strategy.prepare(ema_model, is_rlhf=True)
            del ema_model._offload
        else:
            self.ema_model = None

    def prepare_datasets(self):
        strategy = self.strategy
        args = self.strategy.args

        # prepare datasets
        prompts_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            return_eval=False,
        )
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
        prompts_dataset = PromptDataset(prompts_data, self.tokenizer, strategy, input_template=args.input_template)
        self.prompts_dataloader = strategy.setup_dataloader(prompts_dataset, args.micro_rollout_batch_size, True, True)

        if args.pretrain_data:
            pretrain_data = blending_datasets(
                args.pretrain_data,
                args.pretrain_data_probs,
                strategy,
                args.seed,
                return_eval=False,
            )
            pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len

            pretrain_dataset = SFTDataset(
                pretrain_data.select(range(min(len(pretrain_data), args.max_epochs * len(prompts_dataset)))),
                self.tokenizer,
                pretrain_max_len,
                strategy,
                pretrain_mode=True,
            )

            self.pretrain_dataloader = itertools.cycle(
                iter(
                    strategy.setup_dataloader(
                        pretrain_dataset,
                        args.micro_train_batch_size,
                        True,
                        True,
                        pretrain_dataset.collate_fn,
                    )
                )
            )
        else:
            self.pretrain_dataloader = None

    def max_steps(self):
        """Return the maximum number of steps."""
        return self.max_steps

    def fit(
        self,
        initial_model: ray.actor.ActorHandle,
        reward_model: List[ray.actor.ActorHandle],
        remote_rm_url: List[str] = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        vllm_engines: List[ray.actor.ActorHandle] = None,
    ):
        """Train actor model with prompt datasets."""
        strategy = self.strategy
        args = self.strategy.args

        # configure Trainer
        trainer = ActorReinforceTrainer(
            strategy,
            self.actor,
            reward_model,
            initial_model,
            remote_rm_url=remote_rm_url,
            ema_model=self.ema_model,
            actor_optim=None,
            actor_scheduler=self.actor_scheduler,
            reward_fn=reward_fn,
            vllm_engines=vllm_engines,
            max_epochs=args.max_epochs,
            micro_train_batch_size=args.micro_train_batch_size,
            micro_rollout_batch_size=args.micro_rollout_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            tokenizer=self.tokenizer,
            prompt_max_len=args.prompt_max_len,
            eps_clip=args.eps_clip,
            gamma=args.gamma,
            lambd=args.lambd,
            init_kl_coef=args.init_kl_coef,
            kl_target=args.kl_target,
            ema_beta=0.992,
            ptx_coef=args.ptx_coef,
            max_norm=args.max_norm,
            # fro GPT generation
            do_sample=True,
            max_new_tokens=args.generate_max_len,
            max_length=args.max_len,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            tokenizer_reward=self.tokenizer_reward,
            use_mcts = args.use_mcts,
            use_vinevalue = args.use_vinevalue,
            max_nodes = args.max_nodes,
            max_node_per_depth = args.max_node_per_depth,
            max_time_use = args.max_time_use,
            step_level_norm = args.step_level_norm,
            random_pick = args.random_pick,
            use_orm_reward = args.use_orm_reward,
            select_correct_leaf = args.select_correct_leaf,
            use_chain_reward = args.use_chain_reward,
            use_state_value_reward = args.use_state_value_reward,
            use_value_only = args.use_value_only,
            use_pure_RM = args.use_pure_RM,
            first_token_temperature = args.first_token_temperature,
            use_pure_binary = args.use_pure_binary,
            use_entropy_tree = args.use_entropy_tree,
            average_one_generation = args.average_one_generation,
            advantage_mix_allancestor = args.advantage_mix_allancestor,
            use_weighted_value = args.use_weighted_value,
            balance_ratio = args.balance_ratio,
            use_all_terminals = args.use_all_terminals,
            m = args.m,
            n = args.n,
            l = args.l,
            t = args.t,
            system_prompt = args.system_prompt,
            a_coeff = args.a_coeff,
            b_mean = args.b_mean,
        )

        trainer.fit(self.prompts_dataloader, self.pretrain_dataloader, args)

    def save_model(self):
        args = self.strategy.args

        # save model checkpoint after fitting on only rank0
        self.strategy.save_model(
            self.ema_model if args.enable_ema else self.actor,
            self.tokenizer,
            args.save_path,
        )
