# 1. Video Tokenizer

The first question we are going to answer in this repository is **how to represent a video frame inside a world model**. 

In language models, we convert text into tokens i.e. discrete integers that the model can process. A sentence like "Hello world" becomes something like `[15496, 995]`. This works because language is inherently discrete and can be separated into [multiple chunks or components](https://tiktokenizer.vercel.app/).

But video is continuous. A single 128×128 RGB frame has 128\*128\*3 = 49,152 floating point values. If we were to train a model naively on each frame's raw data, not only the model will blow up in terms of size (imagine a how much weight it would take for a 128x128x3 vector to connect to 1000 neuron) but it will fail to learn the concepts within an image (since the number of dimensions is too vast to represent).

How do we represent a video in a form that a model can process? We briefly talked through this in the introduction of this repository: **with discrete tokens or a continuous latent vector**.

In this guide, we'll go through the former method, discrete tokens, with a **video tokenizer**.

## The Problem

To briefly recap, we want to build a world model that predicts future video frames, but feeding raw pixels to a transformer is problematic:

1. **Too high dimensional**: A 128×128×3 frame = 49,152 values. Processing 4 frames means 196,608 values per sample.
2. **No semantic structure**: Pixels don't capture meaning nor timing i.e. a slight shift in lighting changes all pixel values but the scene is the same, the scene is the same but the time is different.

Assuming you have already worked with [transformers](https://jalammar.github.io/illustrated-transformer/), take 5 minutes before continuing to form research questions: how would you tackle these problems yourself, what kind of questions do you need to answer to solve these problems. 

Out of first principles, to encode and decode a video frame through tokens, several questions come to mind:

1. **How do we reduce the dimensionality?** We can't process 49,152 values per frame efficiently.
2. **How do we preserve position information?** A patch at the top-left should "know" it's at the top-left.
3. **How do we learn relationships between parts of a frame?** Pixels in a patch need to "know" about other patches.
4. **How do we learn relationships across time?** Frame 3 should "know" what happened in frames 1 and 2.
5. **How do we discretize?** Continuous values need to become discrete tokens.
6. **How do we reconstruct pixels from tokens?** We need to decode back to frames for training.

## The Solution 

### 1. How do we reduce the dimensionality?

Inside the 2021 hallmark paper of computer vision, ["An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale"](https://arxiv.org/abs/2010.11929), Dosovitskiy et al introduced a novel method to use a transformer for images.

Instead of processing individual pixels, they divided each frame into patches. A 128×128 frame would be divided into 256 8x8 patches (128^2/8^2=256). Each patch is then projected to a 128-dimensional vector.

The intuition: a small patch of an image (8×8 = 64 pixels) can be summarized by a single vector that captures "what's in this patch." Looking into the code below, we can understand how this process is implemented.

From [models/patch_embed.py](models/patch_embed.py):

```python
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128, patch_size=8, frame_size=128):
        # This is the key of understanding how patch embedding works.
        # We use a Conv2D with kernel=stride=patch_size
        # This extracts non-overlapping patches in an image and projects them in one step to our desired dimensions.
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: (B, T, C, H, W) - batch, time, channels, height, width
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.proj(x)  # (B*T, E, H/P, W/P) = (B*T, 128, 16, 16)
        x = x.flatten(start_dim=2).transpose(1, 2)  # (B*T, 256, 128)
        return x.view(B, T, 256, 128)  # (B, T, N, E)
```

The math here is simple. We slide a convolution kernel across each channel in an image then combine the results to create a single channel in the output. Since we want to have 256 dimensions or channels per patch, we would use 256\*3=768 kernels. A Conv2D with `kernel_size=stride=patch_size` can extract non-overlapping patches and project them in a single operation. 

We go from an image of the following dimension (128, 128, 3) to (16, 16, 128) or (256, 128).

**Add a visualiaztion here. On the left, the input image. On the right, one channel of the output image.**

To learn more about the intuition of convolution, I would recommend not going into the Convolutional Neural Network itself, but understand [its mathematical concept](https://betterexplained.com/articles/intuitive-convolution/). Christopher Olah also has some [great visualizations on CNN](https://colah.github.io/).

### 2. How do we preserve position information?

Before giving the compressed video frame to a transformer, we need to think about what the frame already has in terms of information. It might be able to contain contextual information such as this patch contains a leaf, but it doesn't contain where is that leaf in the frame.

So we need to add some sort of positional encoding. 

For convenience, let's say our image is the following flat vector [0, 0, 0, 0]. Naively, we can add the index of position directly to the vector i.e. [0,1,2,3]. However, as the size of image increase, we would need to use indexes to the length of 256\*128-1=32767. This would pose many problems e.g exploding gradients to scale the distribution of position to another different distribution later in the pipeline.

In the most important paper of modern AI, ["Attenttion Is All You Need"](https://arxiv.org/html/1706.03762v7), Vaswani et al introduced the sinusoidal encoding:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

pos: position of token (in our case, position of the patch)
i: embedding index
d: embedding dimension
```

If you have taken a bit of signal processing, you can see what's the inspiration of this. By using different frequencies, they manage to capture different positions or scales of signal while limiting it to a -1 to 1 range. Low frequencies distinguish "left vs right", while high frequencies distinguish "this pixel vs neighboring pixel."

**Add a visulization here of the model output**

From [models/positional_encoding.py](models/positional_encoding.py):

```python
class SpatioTemporalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim=128, grid_size=16, max_frames=32):
        # Split embedding into spatial (x, y) and temporal (t)
        # 2/3 for spatial, 1/3 for temporal
        self.spatial_dim = (embed_dim * 2) // 3  # ~85 dims for x,y
        self.temporal_dim = embed_dim - self.spatial_dim  # ~43 dims for t

        self.pe_x = SinusoidalPE(embed_dim=self.spatial_dim // 2, max_len=grid_size)  # x position
        self.pe_y = SinusoidalPE(embed_dim=self.spatial_dim // 2, max_len=grid_size)  # y position
        self.pe_t = SinusoidalPE(embed_dim=self.temporal_dim, max_len=max_frames)       # time position

    def forward(self, x):
        # Add position info to each patch embedding
        # x: (B, T, N, E) -> same shape, but now encodes position
        x[:, :, :, :spatial_dim] += spatial_encoding  # x, y position
        x[:, :, :, spatial_dim:] += temporal_encoding  # t position
        return x
```

In the classical paper, sinusoidal encoding was used to encode position for one dimension sequence of tokens. However, in a video, we have 3 dimensions: width, height, time. How do we make it work?

The trick here is to divide embedding into 3 parts `128//3`. The first part is to encode x position, second for y position, and third for time.

For example, to encode x position, we would use a sinusoidal encoding with an embedding dimension of 128//3=42 and position from 0 to 15. To encode time, we would use a sinusoidal encoding with an embedding dimension of 128-42\*2=44 and position of the number of frames we include per sample.

We then combine these encodings to a single embedding and add it to each patch.

### 3. How do we learn relationships between parts of a frame?

Now we have 256 patch embeddings per frame. Even though each patch knows where it is, they were processed independently. The top-left patch doesn't know anything about the bottom-right patch. We need them to communicate because if a patch represents a wall then it's also very likely that its surrounding patches also represent a wall. Capturing the spatial relationship is crucial to representing an frame.

This is what **self-attention** does. Each patch looks at all other patches and computes a weighted sum based on relevance. "How much should I pay attention to each other patch?"

Since the amount of material on attention is (abundant)[https://jalammar.github.io/illustrated-transformer/], I'm not going to write about it anymore. However, it's important to note that unlike attention in LLM, our spatial attention has no causal mask. 

The intuition behind this is simple. In a sentence, the next word is dependent on the previous word. For the prefill and training stage to work in an LLM, we need to introduce a causal mask so that future tokens won't affect past tokens. However, such relation doesn't really exist in a single image. One part of the image attends to all other part of the image. After all, they are captured at the same time.

From [models/st_transformer.py](models/st_transformer.py):

```python
class MultiHeadAttention(nn.Module):
    def forward(self, x):
        # x: (B, N, E) - batch, number of patches, embedding dim

        # Project to queries, keys, values
        q = self.q_proj(x)  # What am I looking for?
        k = self.k_proj(x)  # What do I contain?
        v = self.v_proj(x)  # What information do I have?

        # Attention scores: how relevant is each position to each other position?
        # scores[i,j] = how much should position i attend to position j
        scores = (q @ k.transpose(-2, -1)) / sqrt(head_dim)
        attn_weights = softmax(scores, dim=-1)

        # Weighted sum of values
        output = attn_weights @ v
        return output
```

For **spatial attention**, we process each frame independently. The 256 patches within a frame attend to each other:

```python
# Reshape: (B, T, N, E) -> (B*T, N, E)
# Now each of the B*T frames is a separate sequence of N=256 patches
x = x.view(B * T, N, E)
x = self.spatial_attention(x)  # Patches attend to each other within frame
x = x.view(B, T, N, E)
```

### 4. How do we learn relationships across time?

If you read the last part and ask "What about temporal attention? You must need a causal mask for that", you are right.

Spatial attention lets patches within a frame communicate. But frames also need to communicate with each other. "What happened before me?" It's important to point out that in a world model video encoder, we don't just encode a single frame but also past frames. This contains crucial information, for example the speed of an object.

For **temporal attention**, we flip the axes. Instead of 256 patches attending within a frame, we have each patch position attending across T frames:

```python
# Reshape: (B, T, N, E) -> (B*N, T, E)
# Now each of the B*N patch positions is a separate sequence of T frames
x = x.permute(0, 2, 1, 3).view(B * N, T, E)
x = self.temporal_attention(x, mask=causal_mask)  # Frames attend across time
x = x.view(B, N, T, E).permute(0, 2, 1, 3)
```

As mentioned before, temporal attention is **causal**. We enforce this with a mask:

```python
def create_causal_mask(seq_len):
    # mask[i,j] = True means position i CANNOT attend to position j
    # Upper triangular = can't see the future
    #     [0, 1, 1, 1]   Frame 0 can only see frame 0
    #     [0, 0, 1, 1]   Frame 1 can see frames 0, 1
    #     [0, 0, 0, 1]   Frame 2 can see frames 0, 1, 2
    #     [0, 0, 0, 0]   Frame 3 can see all frames
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
```

### 5. How do we combine spatial and temporal attention?

We combine spatial attention + temporal attention + a feed-forward network into a single block, then stack multiple blocks. This is called a SpatioTemporalTransformer.

**Important clarification**: `SpatioTemporalTransformer` is just a stack of self-attention blocks. It is NOT an encoder-decoder transformer with cross-attention. We'll use the same structure in both our encoder and decoder (with different weights and different causal settings).

```python
class SpatioTemporalBlock(nn.Module):
    def forward(self, x, causal_mask):
        # x: (B, T, N, E)

        # 1. Spatial attention (within each frame)
        x = x + self.spatial_attention(x)

        # 2. Temporal attention (across frames, causal)
        x = x + self.temporal_attention(x, mask=causal_mask)

        # 3. Feed-forward network (process each token independently)
        x = x + self.feed_forward(x)

        return x

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, num_blocks=4, causal_temporal=True):
        self.blocks = [SpatioTemporalBlock() for _ in range(num_blocks)]
        self.causal_temporal = causal_temporal

    def forward(self, x):
        mask = create_causal_mask(T) if self.causal_temporal else None
        for block in self.blocks:
            x = block(x, mask)
        return x
```

If you decide to look into the [source](models/st_transformer.py), you'll find me using a SwiGLU FNN instead of nn.Linear for FNN. The reason is that it is just better thanks to divine benevolence [iykyk](https://arxiv.org/abs/2002.05202).

Jokes aside, TinyWorlds' implementation of genie leveraged this so I wanted to write one from scratch to understand how it works. Big shout out to [TinyWorlds](https://github.com/AlmondGod/tinyworlds).

### 6. How do we discretize?

Let's recap, after the patch embedding, position encoding, and attention, we still have a tensor of shape (256, 128) to represent a frame. And if your intuition serves you right, you would ask "isn't this just a continuous embedding of each patch".

We need discrete tokens.

The classic approach is [**VQ-VAE**](https://arxiv.org/abs/1711.00937): learn a codebook of vectors, find the nearest neighbor for each embedding. But VQ-VAE suffers from codebook collapse (some tokens never get used).

**FSQ** is simpler. We just round each dimension to a finite set of values. 

Let's say we want our patch to be represented by a token, which is represented by a code in a codebook. FSQ takes our tensor (256, 128), turns it to (256, latent\_dim) with a FNN. Now, each patch contains latent\_dim values.

These values are normalized to -1 to 1 range. We divide this range into n bins, then convert each value to the index of the bin it's in.

For example, if we have `latent_dim=3` and `num_bins=3`, a vector of [0.1, 0.4, 0.8] will be turned to [0, 1, 2]. The vector is then treated as a n-base number and converted to decimal: 0\*1 + 1\*3 + 2\*9 = 21.

From [models/fsq.py](models/fsq.py):

```python
class FiniteScalarQuantizer(nn.Module):
    def __init__(self, latent_dim=5, num_bins=4):
        # codebook_size = num_bins^latent_dim = 4^5 = 1024 tokens
        self.latent_dim = latent_dim
        self.num_bins = num_bins

    def forward(self, z):
        # z: (B, T, N, D) where D = latent_dim = 5

        # Step 1: Bound values to [-1, 1]
        z_bounded = torch.tanh(z)

        # Step 2: Scale to [0, num_bins-1] and round
        z_scaled = (z_bounded + 1) / 2 * (self.num_bins - 1)
        z_rounded = torch.round(z_scaled)  # Values in {0, 1, 2, 3}

        # Step 3: Straight-through estimator (STE)
        # Forward: use rounded values
        # Backward: pretend rounding didn't happen (gradients flow through)
        z_q = z_scaled + (z_rounded - z_scaled).detach()

        # Step 4: Convert 5 values to single index
        # [2, 1, 0, 3, 1] -> 2*1 + 1*4 + 0*16 + 3*64 + 1*256 = 454
        indices = (z_rounded * self.basis).sum(dim=-1)

        return z_q, indices  # indices are integers in [0, 1023]
```

With `latent_dim=5` and `num_bins=4` in the above snippet, we get 4^5 = 1024 possible tokens.

### 7. How do we reconstruct pixels from tokens?

We have covered how to turn a frame into a token, but how can we train the model to do such thing. As mentioned before, we treat this as an autoencoder. We have built the encoder, and now we need to build the decoder.

The model is then trained by minimizing the Mean Squared Error loss (or L2 loss) between the original image and reconstructed image.

But how can we go from tokens to pixels?. Step-by-step, we can imagine the process as following:

1. Project from latent dim (5) back to embedding dim (128)
2. Add positional encoding.
3. Add attention through SpatioTemporalTransformer (without the causal mask, since at this step we have all the images already)
4. Convert patch embeddings back to pixels.

While the first 3 steps is straight forward, the tricky part is how to upscale from the patches to a full image. We have 256 embeddings of size 128 and need a 128×128×3 image.

In a naive approach, we can reshape each embedding to 8×8×3 pixels through deconvolution. But this creates [blocky artifacts at patch boundaries](https://distill.pub/2016/deconv-checkerboard/) because convolution kernels cause uneven overlap between patches.

To solve this, Shi et al introduced [**pixel shuffle**](https://arxiv.org/abs/1609.05158). An 8x8 patch is supposed to have 192 values. At the moment, we have 128 values. What if we project it through a linear layer from 128 to 192, then rearrange the 192 values to the patch's rgb values.

From [models/patch_embed.py](models/patch_embed.py):

```python
class PatchUnembedding(nn.Module):
    def __init__(self, embed_dim=128, out_channels=3, patch_size=8):
        # Project each patch embedding to C * P^2 values
        self.proj = nn.Linear(embed_dim, out_channels * patch_size**2)  # 128 -> 192

        # PixelShuffle: (*, C*r^2, H, W) -> (*, C, H*r, W*r)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=patch_size)

    def forward(self, x):
        # x: (B, T, N, E) = (B, T, 256, 128)
        x = self.proj(x)  # (B, T, 256, 192)

        # Reshape to 16x16 grid with 192 channels
        x = x.view(B*T, 16, 16, 192).permute(0, 3, 1, 2)  # (B*T, 192, 16, 16)

        # Pixel shuffle: (B*T, 192, 16, 16) -> (B*T, 3, 128, 128)
        x = self.pixel_shuffle(x)

        return x.view(B, T, 3, 128, 128)
```

The network learns to distribute pixel values across channels in a way that produces smooth reconstructions.

## The Architecture

Now that we've explained each component, here's how they fit together:

```
                              ENCODER
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Video Frames ──► Patch Embed ──► + Pos Enc ──► ST-Transformer ─┼──► Linear ──► FSQ
│  (B,T,3,128,128)  (B,T,256,128)   (B,T,256,128)  (B,T,256,128)  │   (B,T,256,5)   ▼
│                                   [solution 5]   [solutions 2-4]│               ┌────────┐
└─────────────────────────────────────────────────────────────────┘               │ Tokens │
    [solution 1]                                                                  │(B,T,256)│
                              DECODER                                             └────┬───┘
┌─────────────────────────────────────────────────────────────────┐                    │
│                                                                 │    [solution 6]    ▼
│  Reconstructed ◄── Pixel Shuffle ◄── ST-Transformer ◄── + Pos ─┼─◄── Linear ◄── z_q
│  (B,T,3,128,128)   (B,T,256,128)     (B,T,256,128)       Enc    │   (B,T,256,128)
│                    [solution 7]      [non-causal]               │
└─────────────────────────────────────────────────────────────────┘

Training: minimize MSE(Video Frames, Reconstructed)
```

**Why do encoder and decoder both have their own ST-Transformer?**

This is an **autoencoder**, not a seq2seq model:

```
Seq2seq (e.g., translation):
  Encoder: "Hello" → [hidden states]
                          ↓ cross-attention (decoder looks at encoder output)
  Decoder: [hidden states] → "Bonjour"

Autoencoder (what we're doing):
  Encoder: frames → [bottleneck: discrete tokens]
                          ↓ just pass the tokens through (no cross-attention)
  Decoder: [tokens] → reconstructed frames
```

There's no cross-attention. The encoder and decoder only communicate through the **bottleneck** (the discrete tokens). Both use the same `SpatioTemporalTransformer` class but with:

| | Encoder's Transformer | Decoder's Transformer |
|---|---|---|
| Temporal attention | Causal (can't see future) | Non-causal (sees all) |
| Weights | Learned independently | Learned independently |
| Cross-attention | None | None |

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
| D | Latent dimensions (FSQ) | 5 |
| L | Bins per dimension (FSQ) | 4 |

## Usage

```bash

# Train with dummy data (sanity check)
uv run python train.py --use-dummy-data --num-epochs 10

# Train with video folder
uv run python train.py --data-path /path/to/videos --data-type folder

# Validate and visualize
uv run python validate.py --checkpoint checkpoints/best_model.pt --save-images
```

## What to look for

During training:
- **Loss should decrease** - if it doesn't, learning rate might be wrong
- **Codebook usage should be high** - all 1024 tokens should eventually be used
- **Validation should track training** - if val loss diverges, you're overfitting

During validation:
- **PSNR > 25 dB** - reasonable quality
- **PSNR > 30 dB** - good quality
- **Codebook usage > 90%** - efficient use of vocabulary

## Files

```
1.video-tokenizer/
├── README.md                           # This file
├── config.py                           # Hyperparameters
├── data_utils.py                       # Dataset classes
├── train.py                            # Training loop
├── validate.py                         # Evaluation
├── models/
│   ├── video_tokenizer.py              # Main model
│   ├── fsq.py                          # Finite Scalar Quantization
│   ├── st_transformer.py               # Spatio-Temporal Transformer
│   ├── patch_embed.py                  # Patch Embedding
│   └── positional_encoding.py          # Positional Encoding
├── checkpoints/                        # Saved models
└── data/                               # Datasets
```

## Run Log

To validate the model, I used egocentric videos (`factory001/worker001/part000.tar`) from [BuildAI's egocentric video dataset](https://huggingface.co/datasets/builddotai/Egocentric-100K).

There are a total of 77 videos at 30fps totaling a duration of 13789s ~= 3.83hr. The resolution for each video is 456x256, which is then resized to 128x128 using opencv.resize's default interpolation.

For compute, I used a spot L4 GPU (24GB VRAM) isntance on GCP with 16 cpu cores. The following parameters were used:

```
  uv run ./1.video-tokenizer/train.py \
    --data-path ./1.video-tokenizer/data/ \
    --data-type folder \
    --batch-size 48 \
    --embed-dim 256 \
    --num-blocks 6 \
    --num-epochs 3 \
    --num-workers 16
```

The output of such command is here for reference:

```
Using folder data from ./1.video-tokenizer/data/

Setting up data...
Found 77 videos in ./1.video-tokenizer/data/
Total clips: 103402
Train samples: 93061
Val samples: 10341

Creating model...
Model parameters: 12,743,365
Codebook size: 1024
Device: cuda
```

![Training run](../assets/1.video-tokenizer/training_run.png)

It costs me roughly 2.1 hours to train. The checkpoints can be found in `checkpoints/`.

For validation, the following command is used:

```
uv run python validate.py --checkpoint checkpoints/best_model.pt --data-path ./data --data-type folder --save-images --nubatches 5 --num-samples 5
```

Reconstruction quality on real video frames from the validation set (PSNR ~22 dB):

| | | | | |
|---|---|---|---|---|
| ![Sample 1](../assets/1.video-tokenizer/real-decode/sample_1.png) | ![Sample 2](../assets/1.video-tokenizer/real-decode/sample_2.png) | ![Sample 3](../assets/1.video-tokenizer/real-decode/sample_3.png) | ![Sample 4](../assets/1.video-tokenizer/real-decode/sample_4.png) | ![Sample 5](../assets/1.video-tokenizer/real-decode/sample_5.png) |

What happens when we feed random noise through the tokenizer? The decoder produces blurry, averaged outputs since noise doesn't map to meaningful tokens:

| | | | | |
|---|---|---|---|---|
| ![Sample 1](../assets/1.video-tokenizer/noise-decode/sample_1.png) | ![Sample 2](../assets/1.video-tokenizer/noise-decode/sample_2.png) | ![Sample 3](../assets/1.video-tokenizer/noise-decode/sample_3.png) | ![Sample 4](../assets/1.video-tokenizer/noise-decode/sample_4.png) | ![Sample 5](../assets/1.video-tokenizer/noise-decode/sample_5.png) |

### Using the Checkpoints

The checkpoint files are stored using Git LFS. To download them:

```bash
# Install git-lfs if you haven't already
# Ubuntu/Debian: sudo apt install git-lfs
# macOS: brew install git-lfs

# Initialize git-lfs and pull the files
git lfs install
git lfs pull
```

**Security note:** PyTorch checkpoint files (`.pt`) use pickle serialization, which can execute arbitrary code when loaded. Only load checkpoints from sources you trust. The code uses `weights_only=False` in `torch.load()` to load the config object stored alongside the weights.

You can validate and visualize the model with the following command:

```bash
uv run python validate.py --checkpoint checkpoints/best_model.pt --data-path ./data --save-images
```

## Disclaimer/Improvements

This implementation is not built for speed. A few improvements come to mind:

[ ] Pre-process Video: The Dataloader decodes videos on the fly. This bottlenecks the training as the GPU needs to wait for the CPU on every step.
[ ] Kernel Optimization: We still use vanilla attention and kernels. Speed can improve significantly by kernels such as FlashAttention.

I recommend implementing these yourself to learn.

## What's next?

Now that we can tokenize video frames into discrete tokens, the next question is: **how do we predict future tokens?**

Given the tokens for frames 1-4 and an action, can we predict the tokens for frame 5? This is what the dynamics model does - covered in `2.dynamics-model/`.

## References

- [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391)
- [Language Model Beats Diffusion - Tokenizer is Key](https://arxiv.org/abs/2310.05737) (FSQ paper)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) (ViT)
