"""
Base class for Context-Aware Recommender System (CARS) metrics.

All CARS metrics accumulate predictions and context batch-by-batch
(following the TorchMetrics pattern used in WarpRec) and compute
the final score in compute().

Context tensors:
    - item_context_lookup: [num_items, num_features]  (static per item)
    - user_context_lookup: [num_users, num_features]  (mode per user)

Both are stored as buffers (like feature_lookup in SRecall) and passed
via common_params from the Evaluator (populated from dataset.get_stash()).

FIX: Use tensor-based accumulation (torch.cat) instead of list-based
add_state, which TorchMetrics does not track correctly for appending.
"""

from abc import abstractmethod
from typing import Any, Optional, Set

import torch
from torch import Tensor
from warprec.evaluation.metrics.base_metric import BaseMetric
from warprec.utils.enums import MetricBlock


class BaseCARSMetric(BaseMetric):
    """Base class for CARS metrics.

    Accumulates top-k item indices and user indices batch-by-batch
    using tensor concatenation (not list), then computes context-aware
    metrics in compute().

    Args:
        k (int): Cutoff.
        num_users (int): Number of users.
        num_items (int): Number of items.
        item_context_lookup (Tensor): [num_items, num_features] item context tensor.
        user_context_lookup (Tensor): [num_users, num_features] user context tensor.
        context_feature_weights (Optional[Tensor]): [num_features] IDF weights.
        feature_groups (Optional[dict]): Feature group indices for CGB.
        dist_sync_on_step (bool): TorchMetrics distributed sync.
    """

    _REQUIRED_COMPONENTS: Set[MetricBlock] = {
        MetricBlock.TOP_K_INDICES,
        MetricBlock.TOP_K_VALUES,
        MetricBlock.VALID_USERS,
        MetricBlock.TOP_K_BINARY_RELEVANCE,
    }

    def __init__(
        self,
        k: int,
        num_users: int,
        num_items: int,
        item_context_lookup: Tensor,
        user_context_lookup: Tensor,
        context_feature_weights: Optional[Tensor] = None,
        feature_groups: Optional[dict] = None,
        dist_sync_on_step: bool = False,
        **kwargs: Any,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.k = k
        self.num_users = num_users
        self.num_features = item_context_lookup.shape[1]
        self.feature_groups = feature_groups or {}

        # Register context tensors as buffers (moved to device automatically)
        self.register_buffer("item_context_lookup", item_context_lookup.float())
        self.register_buffer("user_context_lookup", user_context_lookup.float())

        # IDF weights: ones by default (unweighted), or provided externally
        if context_feature_weights is None:
            context_feature_weights = torch.ones(self.num_features)
        self.register_buffer("feature_weights", context_feature_weights.float())

        # Accumulators: tensor-based (cat), NOT list-based add_state.
        # TorchMetrics does not correctly track list mutation via append.
        self.add_state(
            "all_top_k_indices",
            default=torch.zeros(0, k, dtype=torch.long),
            dist_reduce_fx="cat",
        )
        self.add_state(
            "all_user_indices",
            default=torch.zeros(0, dtype=torch.long),
            dist_reduce_fx="cat",
        )
        self.add_state(
            "all_valid_users",
            default=torch.zeros(0, dtype=torch.float),
            dist_reduce_fx="cat",
        )

    def update(self, preds: Tensor, **kwargs: Any) -> None:
        """Accumulate batch predictions via tensor concatenation."""
        top_k_indices = kwargs.get(f"top_{self.k}_indices")  # [B, k]
        user_indices = kwargs.get("user_indices")              # [B]
        valid_users = kwargs.get("valid_users")                # [B]

        if top_k_indices is None or user_indices is None:
            return

        device = self.item_context_lookup.device
        top_k_indices = top_k_indices.to(device)
        user_indices = user_indices.to(device)

        self.all_top_k_indices = torch.cat(
            [self.all_top_k_indices, top_k_indices], dim=0
        )
        self.all_user_indices = torch.cat(
            [self.all_user_indices, user_indices], dim=0
        )

        if valid_users is not None:
            valid_users = valid_users.float().to(device)
            self.all_valid_users = torch.cat(
                [self.all_valid_users, valid_users], dim=0
            )

    def compute(self) -> dict:
        """Compute CARS metric over all accumulated batches."""
        print(f"[COMPUTE {self.__class__.__name__} k={self.k}] shape={self.all_top_k_indices.shape}")
        if self.all_top_k_indices.shape[0] == 0:
            return {self._metric_name(): 0.0}

        device = self.item_context_lookup.device
        top_k = self.all_top_k_indices.to(device)    # [N, k]
        users = self.all_user_indices.to(device)      # [N]

        if self.all_valid_users.shape[0] > 0:
            valid = self.all_valid_users.to(device)   # [N]
        else:
            valid = torch.ones(top_k.shape[0], device=device)

        item_ctx = self._get_item_ctx(top_k)              # [N, k, F]
        user_ctx = self._get_user_ctx(users)              # [N, F]
        match = self._context_match(user_ctx, item_ctx)   # [N, k, F]

        if self.all_user_indices.shape[0] > 0:
            # Print the first user of the first batch
            print(f"DEBUG REAL MATCH (User {users[0]}): {match[0].mean(dim=-1)}")

        scores = self._compute_cars(match, user_ctx, item_ctx, valid)
        print(f"[RESULT k={self.k}] {scores}")
        return scores

    def _get_item_ctx(self, top_k_indices: Tensor) -> Tensor:
        """[N, k] -> [N, k, F]"""
        return self.item_context_lookup[top_k_indices]

    def _get_user_ctx(self, user_indices: Tensor) -> Tensor:
        """[N] -> [N, F]"""
        return self.user_context_lookup[user_indices]

    def _context_match(self, user_ctx: Tensor, item_ctx: Tensor) -> Tensor:
        """
        Per-feature match between user and item context.
        Args:
            user_ctx: [N, F]
            item_ctx: [N, k, F]
        Returns:
            match: [N, k, F] — 1.0 where feature matches, 0.0 otherwise
        """
        return (item_ctx == user_ctx.unsqueeze(1)).float()

    @abstractmethod
    def _compute_cars(
        self,
        match: Tensor,
        user_ctx: Tensor,
        item_ctx: Tensor,
        valid: Tensor,
    ) -> dict:
        """Subclasses implement the actual CARS metric here."""

    @abstractmethod
    def _metric_name(self) -> str:
        """Return the metric name string, e.g. 'CS'."""

    def _weighted_mean(self, scores: Tensor, valid: Tensor) -> float:
        """Weighted average over valid users."""
        valid = valid.to(scores.device)
        n_valid = valid.sum().clamp(min=1)
        return (scores * valid).sum().item() / n_valid.item()