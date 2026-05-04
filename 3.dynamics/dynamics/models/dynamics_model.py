"""
Token Dynamics Model implementation from first principles.

The model learns the world-model transition:
    tokens_t + action_t -> tokens_t+1

It does not predict pixels directly. Pixels are handled by the video tokenizer.
This model predicts the next discrete visual token for every patch.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from video_tokenizer import (
    SpatioTemporalPositionalEncoding,
    SpatioTemporalTransformer,
)


class TokenDynamicsModel(nn.Module):
    """
    Predict next visual tokens from current visual tokens and latent actions.

    Args:
        vocab_size: Number of visual tokens from the video tokenizer (V)
        num_patches: Number of patch tokens per frame (N)
        action_dim: Dimension of quantized latent action vectors (A)
        n_actions: Optional discrete action vocabulary size
        embed_dim: Transformer embedding dimension (E)
        num_heads: Number of attention heads
        num_blocks: Number of spatio-temporal transformer blocks
        grid_size: Number of patches per row/column
        max_frames: Maximum temporal context length
        dropout: Dropout probability
        use_adaptive_conditioning: Whether to also condition transformer layers
            on the mean action vector. Additive per-step conditioning is always used.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        num_patches: int = 256,
        action_dim: int = 3,
        n_actions: int = 8,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_blocks: int = 4,
        grid_size: int = 16,
        max_frames: int = 32,
        dropout: float = 0.0,
        use_adaptive_conditioning: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_patches = num_patches
        self.action_dim = action_dim
        self.n_actions = n_actions
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.use_adaptive_conditioning = use_adaptive_conditioning

        # Visual token ids are integers in [0, vocab_size).
        # token_embed: (B, T, N) -> (B, T, N, E)
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Latent actions can be passed either as:
        # - action vectors: (B, T, A), usually quantized values in [-1, 1]
        # - action ids: (B, T), integers in [0, n_actions)
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.action_id_embed = nn.Embedding(n_actions, embed_dim)

        self.pos_encoding = SpatioTemporalPositionalEncoding(
            embed_dim=embed_dim,
            grid_size=grid_size,
            max_frames=max_frames,
        )

        conditioning_dim = action_dim if use_adaptive_conditioning else None
        self.transformer = SpatioTemporalTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            dropout=dropout,
            causal_temporal=True,
            conditioning_dim=conditioning_dim,
        )

        # One classifier per patch token.
        # (B, T, N, E) -> (B, T, N, V)
        self.output_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_size),
        )

    def _embed_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Convert actions to embedding vectors.

        Args:
            actions:
                - (B, T, A) latent action vectors, or
                - (B, T) discrete action ids

        Returns:
            action_embeddings: (B, T, E)
        """
        if actions.dim() == 2:
            # Discrete action id path.
            return self.action_id_embed(actions.long())

        if actions.dim() == 3:
            if actions.shape[-1] != self.action_dim:
                raise ValueError(
                    f"Expected action_dim={self.action_dim}, got {actions.shape[-1]}"
                )
            return self.action_proj(actions.float())

        raise ValueError(
            "actions must have shape (B, T) for ids or (B, T, A) for latent vectors"
        )

    def _action_conditioning(self, actions: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Create one conditioning vector per sample for adaptive normalization.

        Additive conditioning already gives the model per-timestep actions.
        Adaptive conditioning is optional and intentionally coarse: it summarizes
        the action sequence for the transformer blocks.
        """
        if not self.use_adaptive_conditioning:
            return None

        if actions.dim() == 2:
            # Convert ids to a binary-ish scalar summary is not meaningful enough here.
            # Keep adaptive conditioning for latent vectors only.
            return None

        return actions.float().mean(dim=1)

    def forward(self, video_tokens: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Predict next-token logits for each input frame/action pair.

        Args:
            video_tokens: Current visual tokens, shape (B, T, N)
            actions: Actions for each transition, shape (B, T, A) or (B, T)

        Returns:
            logits: Next-token logits, shape (B, T, N, V)
        """
        if video_tokens.dim() != 3:
            raise ValueError("video_tokens must have shape (B, T, N)")

        B, T, N = video_tokens.shape
        if N != self.num_patches:
            raise ValueError(f"Expected {self.num_patches} patches, got {N}")
        if actions.shape[0] != B or actions.shape[1] != T:
            raise ValueError("actions must have the same B and T as video_tokens")

        # x: (B, T, N, E)
        x = self.token_embed(video_tokens.long())
        x = self.pos_encoding(x, add_temporal=True)

        # action_embeddings: (B, T, E) -> (B, T, N, E)
        action_embeddings = self._embed_actions(actions)
        action_embeddings = action_embeddings.unsqueeze(2).expand(-1, -1, N, -1)
        x = x + action_embeddings

        conditioning = self._action_conditioning(actions)
        x = self.transformer(x, conditioning=conditioning)

        logits = self.output_head(x)
        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy over visual token ids.

        Args:
            logits: Predicted token logits, shape (B, T, N, V)
            target_tokens: Target token ids, shape (B, T, N)

        Returns:
            loss: Scalar cross-entropy loss
        """
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_tokens.long().reshape(-1),
        )

    def training_step(
        self,
        video_tokens: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher-forced training step.

        Args:
            video_tokens: Tokenized video clip, shape (B, T, N)
            actions: Actions between frames, shape (B, T-1, A) or (B, T-1)

        Returns:
            loss: Cross-entropy loss
            logits: Predicted logits for tokens[:, 1:], shape (B, T-1, N, V)
        """
        input_tokens = video_tokens[:, :-1].contiguous()
        target_tokens = video_tokens[:, 1:].contiguous()

        if actions.shape[1] != input_tokens.shape[1]:
            raise ValueError(
                "actions must contain one action for each input token frame "
                f"({input_tokens.shape[1]}), got {actions.shape[1]}"
            )

        logits = self(input_tokens, actions)
        loss = self.compute_loss(logits, target_tokens)
        return loss, logits

    @torch.no_grad()
    def predict_next(
        self,
        context_tokens: torch.Tensor,
        context_actions: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = False,
    ) -> torch.Tensor:
        """
        Predict the next frame tokens from a context sequence.

        Args:
            context_tokens: Current token context, shape (B, T, N)
            context_actions: Actions aligned with context tokens, shape (B, T, A) or (B, T)
            temperature: Sampling temperature
            sample: If True, sample from the distribution; otherwise use argmax

        Returns:
            next_tokens: Predicted next frame tokens, shape (B, N)
        """
        logits = self(context_tokens, context_actions)
        next_logits = logits[:, -1] / max(temperature, 1e-6)  # (B, N, V)

        if sample:
            probs = F.softmax(next_logits, dim=-1)
            return torch.multinomial(
                probs.reshape(-1, self.vocab_size),
                num_samples=1,
            ).view(context_tokens.shape[0], self.num_patches)

        return next_logits.argmax(dim=-1)

    @torch.no_grad()
    def rollout(
        self,
        start_tokens: torch.Tensor,
        actions: torch.Tensor,
        temperature: float = 1.0,
        sample: bool = False,
    ) -> torch.Tensor:
        """
        Roll the dynamics model forward from one starting token frame.

        Args:
            start_tokens: Starting frame tokens, shape (B, N)
            actions: Planned actions, shape (B, S, A) or (B, S)
            temperature: Sampling temperature
            sample: If True, sample next tokens; otherwise use argmax

        Returns:
            generated_tokens: Token rollout, shape (B, S+1, N)
        """
        if start_tokens.dim() != 2:
            raise ValueError("start_tokens must have shape (B, N)")

        generated = [start_tokens.long()]
        used_actions = []

        for step in range(actions.shape[1]):
            used_actions.append(actions[:, step])

            context_tokens = torch.stack(generated, dim=1)
            context_actions = torch.stack(used_actions, dim=1)

            next_tokens = self.predict_next(
                context_tokens=context_tokens,
                context_actions=context_actions,
                temperature=temperature,
                sample=sample,
            )
            generated.append(next_tokens)

        return torch.stack(generated, dim=1)


if __name__ == "__main__":
    print("Testing Token Dynamics Model...")

    model = TokenDynamicsModel()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    tokens = torch.randint(0, 1024, (2, 4, 256))
    actions = torch.randint(0, 2, (2, 3, 3)).float() * 2 - 1

    loss, logits = model.training_step(tokens, actions)
    print(f"Input tokens: {tokens.shape}")
    print(f"Actions: {actions.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    rollout = model.rollout(tokens[:, 0], actions)
    print(f"Rollout: {rollout.shape}")
    print("All tests passed!")
