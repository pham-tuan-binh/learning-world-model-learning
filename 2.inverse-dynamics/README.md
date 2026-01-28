# 2. Inverse Dynamics

The biggest problem with training a world model is the lack of action labels for videos. If we retrace the definition of world model from the first article, we need to train a model to predict the future based on current state and action. Getting the current state is easy, it's just the current video frame. But where do we get the actions from?

The process of predicting the next frame is called **dynamics**. Much like dynamics in physics, where it's about how forces interact with each other in a scene. If we apply a force on an object, what would be its velocity? In world models, this question becomes: **if we apply this action to the current state, what would be the next state?**

In robotics, we have **inverse kinematics**, where we have the end pose of a robot's end effector and we need to calculate the joint positions to get there. **Inverse dynamics** concerns calculating the necessary torque and forces of each joint to get a desired motion.

In world models, **inverse dynamics means inferring the action from consecutive frames**. Given frame at time t and frame at time t+1, what action caused this transition?

## The Problem

Consider training a world model on YouTube videos. We have millions of hours of video showing people walking, cars driving, objects moving. But we don't have labels for what "actions" caused these movements. Without action labels, how can we train a model that takes (state, action) → next_state?

The naive approaches don't work well:
1. **Ignore actions entirely**: Just predict next frame from current frame. But then the model can't be controlled, it will hallucinate its own "actions."
2. **Manual labeling**: Have humans label each frame transition with actions. Prohibitively expensive and doesn't scale.
3. **Use controller inputs**: Only works for games/simulations where we can record inputs. Doesn't work for real-world video.

The key insight from [Genie](https://arxiv.org/abs/2402.15391) is elegant: **learn the actions themselves**. If we can train a model to infer what action happened between frames, we can use those inferred actions to train the dynamics model.

## The Questions

Out of first principles, to build an inverse dynamics model, we need to answer:

1. **How can we represent video frames?** We answered this question in the last article.
2. **How can we represent actions?** For games, actions can be inputs from a controller. But in the real world, actions are normally more nuanced.
3. **How do we infer actions from frame pairs?** What kind of architecture look at two frames and outputs an action?
4. **How do we train without ground truth?** We don't have action labels to supervise with.

And then we have to answer practical problems such as:

5. **How do we ensure that actions are meaningful?** Models can easily collapse and ignore actions entirely for next frame prediction.

## The Solutions

### 1. How can we represent video frames?

This problem has been addressed in the last article. In short, we go from a traditional video frame `(H, W, 3)` to a discrete token representation `(P, 1)` where a frame is divided into patches and each patch is represented by a number token, from 0 to 1023.

Practically, this token is represented with a vector of size `(5)` where the value in each dimension is put into 4 bins of values. The total number of tokens is therefore `4^5=1024`.

In code, you'll find the input data of our model to be `(B, T, P, D)` where B is the batch size, T is the number of frames, P is the number of patches and D is the latent dim.

### 2. How can we represent actions?

In video games, actions can be recognized easily as games are played with controllers. For example, moving forward or moving joystick up can be represented as single number which tells the velocity of the player. This is continuous actions. Or we can represent discrete actions such as dpad buttons through quantization or one hot encoding.

In the real world, actions are much more nuanced. As humans, our primary way of describing actions is either through words or demonstration. For example, walk forward 1 meter or jump. We don't have a way to accurately describe one's action. How do we expect computers to understand it?

This is normally solved with [latent actions](https://arxiv.org/abs/2410.11758), which essentially captures the meaning of an action before relying on another model to output accurate actions based on embodiment such as position of a motor.

In our case, we'll use discrete tokens for discrete actions for the sake of simplicity. The technique has already been discussed in the last article in the name of FSQ.

For a code book size of 8 or 8 discrete actions, we should predict a vector of size (3) where each dimension is put into 2 bins. This gives us `2^3=8` codes.

### 3. How can we infer actions from frame pairs?

Let's re-frame our question: if we have 2 video frame, which is each represented as a vector of size `(P, D)`, how can we predict a vector of size `3`. In essence, we have  


### 1. How do we infer actions from frame pairs?

We use an encoder that takes a sequence of frames and outputs one action per frame transition. The architecture:

1. **Patch embedding**: Convert each frame to patch tokens (reusing from video tokenizer)
2. **Spatio-temporal transformer**: Learn relationships within and across frames
3. **Mean pooling**: Aggregate patches into one vector per frame
4. **Action head**: Concatenate adjacent frame vectors and predict action

From [models/latent_action_model.py](models/latent_action_model.py):

```python
class LatentActionsEncoder(nn.Module):
    def forward(self, frames):
        # frames: (B, T, C, H, W)

        # Convert to patch embeddings
        embeddings = self.patch_embed(frames)  # (B, T, N, E)

        # Add positional encoding
        embeddings = self.pos_encoding(embeddings)

        # Apply spatio-temporal transformer
        transformed = self.transformer(embeddings)  # (B, T, N, E)

        # Mean pool over patches (one vector per frame)
        pooled = transformed.mean(dim=2)  # (B, T, E)

        # Infer action from each adjacent pair
        actions = []
        for t in range(seq_len - 1):
            # Concatenate current and next frame features
            combined = torch.cat([pooled[:, t], pooled[:, t + 1]], dim=1)
            action = self.action_head(combined)  # (B, A)
            actions.append(action)

        return torch.stack(actions, dim=1)  # (B, T-1, A)
```

The intuition: by looking at what changed between frames, the model learns to extract the "delta" - the action that caused the change.

### 2. How do we make actions discrete?

We use the same FSQ (Finite Scalar Quantization) from the video tokenizer, but with binary quantization. Each action dimension is rounded to 0 or 1, giving us `2^D` possible discrete actions.

For example, with `D=3` dimensions:
- `2^3 = 8` possible discrete actions
- Action `[0,0,0]` = action 0 (perhaps "do nothing")
- Action `[1,0,0]` = action 1 (perhaps "move left")
- Action `[1,1,1]` = action 7 (perhaps "move diagonally + jump")

The model learns what each action "means" through training - we don't define the semantics, they emerge.

```python
# From models/latent_action_model.py
self.quantizer = FiniteScalarQuantizer(
    latent_dim=self.action_dim,    # D dimensions
    num_bins=2,                     # Binary: {0, 1}
)
# n_actions = 2^action_dim = 2^3 = 8 discrete actions
```

### 3. How do we ensure actions are meaningful?

This is the key challenge. Without any constraint, the decoder might learn to:
- Ignore the action and just copy the input frame
- Predict the "average" next frame regardless of action

**Solution: Aggressive token masking.**

During training, we mask almost all tokens in the decoder input. The decoder only sees:
- The first frame (as an anchor)
- The action conditioning

This forces the action to carry the information about what happened. If the action were meaningless, the decoder couldn't reconstruct the target frame.

```python
class LatentActionsDecoder(nn.Module):
    def forward(self, frames, actions, training=True):
        # ...

        # Mask tokens during training to force reliance on actions
        if training and self.training:
            keep_rate = 0.0  # Mask ALL tokens except first frame
            keep = torch.rand(...) < keep_rate
            keep[:, 0] = True  # Never mask first frame (anchor)
            video_embeddings = torch.where(
                keep, video_embeddings,
                self.mask_token.expand_as(video_embeddings)
            )
```

We also add a **variance penalty** to prevent action collapse (where the encoder predicts the same action for everything):

```python
# Encourage diversity in predicted actions
z_var = action_latents.var(dim=0).mean()
var_penalty = F.relu(self.var_target - z_var)
total_loss = recon_loss + self.var_lambda * var_penalty
```

### 4. How do we train without ground truth?

The training signal is **reconstruction**. If our inferred actions are correct, we should be able to use them to predict the next frame.

```
Training loop:
1. Encoder sees frames [1, 2, 3, 4]
2. Encoder outputs actions [a1, a2, a3] (between each pair)
3. Quantizer discretizes actions
4. Decoder takes frames [1, 2, 3] + actions [a1, a2, a3]
5. Decoder predicts frames [2', 3', 4']
6. Loss = difference between [2', 3', 4'] and actual [2, 3, 4]
```

The key insight: we never need ground truth actions. The model learns to infer actions that are **useful for prediction**. Actions that don't help prediction get zero gradient - they're not learned.

## The Architecture

```
                           ENCODER (Inverse Dynamics)
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  Frames ──► Patch Embed ──► + Pos Enc ──► ST-Transformer ──► Mean Pool  │
│  (B,T,3,H,W)  (B,T,N,E)     (B,T,N,E)    (B,T,N,E)         (B,T,E)      │
│                                                                  │       │
│                                              ┌───────────────────┘       │
│                                              ▼                           │
│                              Action Head: concat(frame_t, frame_t+1)     │
│                                              │                           │
└──────────────────────────────────────────────┼───────────────────────────┘
                                               ▼
                                    Continuous Actions (B, T-1, A)
                                               │
                                               ▼
                                          ┌─────────┐
                                          │   FSQ   │  Binary Quantization
                                          └────┬────┘
                                               ▼
                                    Discrete Actions (B, T-1, A)
                                               │
                           DECODER (Forward Dynamics)
┌──────────────────────────────────────────────┼───────────────────────────┐
│                                              ▼                           │
│  Frames[:-1] ──► Patch Embed ──► + Pos Enc ──► + Action Proj            │
│  (B,T-1,3,H,W)   (B,T-1,N,E)    (B,T-1,N,E)    (B,T-1,N,E)              │
│                                                    │                     │
│                                      [Token Masking during training]     │
│                                                    │                     │
│                                                    ▼                     │
│                                            ST-Transformer                │
│                                                    │                     │
│                                                    ▼                     │
│                                              Frame Head                  │
│                                                    │                     │
│                                                    ▼                     │
│                                         Predicted Frames (B,T-1,3,H,W)   │
└──────────────────────────────────────────────────────────────────────────┘

Training: minimize reconstruction loss + variance penalty
```

## Dimensions Reference

| Symbol | Meaning | Default |
|--------|---------|---------|
| B | Batch size | 8 |
| T | Number of frames | 4 |
| C | Channels (RGB) | 3 |
| H, W | Frame height/width | 128 |
| P | Patch size | 8 |
| N | Patches per frame = (H/P)² | 256 |
| E | Embedding dimension | 128 |
| A | Action dimension | 3 |
| n_actions | Discrete vocabulary = 2^A | 8 |

## Usage

```bash
# Train with dummy data (sanity check)
uv run python train.py --use-dummy-data --num-epochs 10

# Train with video folder
uv run python train.py --data-path /path/to/videos --data-type folder

# Train with more actions (16 = 2^4)
uv run python train.py --data-path /path/to/videos --n-actions 16

# Validate and analyze
uv run python validate.py --checkpoint checkpoints/best_model.pt --save-images
```

## What to Look For

During training:
- **Loss should decrease** - if not, learning rate might be wrong
- **Action variance should stay above target** - if it collapses to 0, increase `var_lambda`
- **Unique actions used** - ideally all `n_actions` get used

During validation:
- **Reconstruction quality** - predicted frames should resemble targets
- **Action diversity** - different transitions should produce different actions
- **Action consistency** - similar transitions should produce similar actions

## Files

```
2.inverse-dynamics/
├── README.md                           # This file
├── config.py                           # Hyperparameters
├── data_utils.py                       # Dataset loading (reuses folder 1)
├── train.py                            # Training loop
├── validate.py                         # Evaluation and visualization
├── models/
│   ├── __init__.py                     # Package exports
│   └── latent_action_model.py          # Encoder + Decoder + Full model
└── checkpoints/                        # Saved models
```

## Reused Components

This folder reuses components from `1.video-tokenizer/`:
- `PatchEmbedding`: Convert frames to patch tokens
- `SpatioTemporalTransformer`: Attention over space and time
- `SpatioTemporalPositionalEncoding`: Position information
- `FiniteScalarQuantizer`: Discretize latents

## What's Next?

Now that we can infer latent actions from video, the next step is to build the **dynamics model**. Given frame tokens + action, predict the next frame's tokens. This completes the world model:

1. **Video Tokenizer** (folder 1): frame → tokens
2. **Inverse Dynamics** (folder 2): frames → actions
3. **Dynamics Model** (folder 3): tokens + action → next tokens

The dynamics model uses the inferred actions to learn controllable video prediction.

## References

- **Genie** - Bruce et al., "Genie: Generative Interactive Environments", Google DeepMind, 2024. [arXiv:2402.15391](https://arxiv.org/abs/2402.15391)
- **GameNGen** - Valevski et al., "Diffusion Models Are Real-Time Game Engines", Google, 2024. [arXiv:2408.14837](https://arxiv.org/abs/2408.14837)
