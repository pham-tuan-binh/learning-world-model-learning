# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an educational repository for exploring world models in robotics. A world model is a prediction model that, given the current state and action, predicts the next state. The focus is specifically on video world models where frames are predicted based on current video frame + action.

The goal is to build intuition about how world models work from first principles, following the approach used in models like Genie, Nvidia Cosmo, and V-JEPA.

## Repository Structure

Each numbered folder (e.g., `1.video-encoder-decoder`) contains a self-contained model implementation with:
- `data/` - Dataset files
- `data_utils.py` - Dataset downloading and loading
- `models/` - Model architecture and component models
- `train.py` - Training script
- `validate.py` - Validation script
- `checkpoints/` - Model checkpoints

Folders are meant to be explored sequentially (1 to n) to build intuition about world model components.

## Key Concepts

The three core questions answered by each implementation:
1. How to represent the video frame (e.g., VAE encoding)
2. How to represent the action (discrete or continuous vectors)
3. Where to get the action (inverse dynamics model from frame pairs, controller overlay extraction, etc.)

## Development Guidelines

### Package Management with uv

Always use `uv` for everything Python-related:
- Run Python scripts: `uv run python script.py` or `uv run train.py`
- Add dependencies: `uv add package_name`
- Add dev dependencies: `uv add --dev package_name`
- Never use `pip install` or `python` directly

### First Principles Approach

- Always reason from first principles when implementing components
- Ask "why" before "how" - understand the purpose before writing code
- Build from scratch unless the component already exists as a well-established Python library
- Prefer implementing core algorithms yourself over using high-level abstractions (e.g., implement attention mechanism instead of using a black-box transformer library)
- Use established libraries only for: numerical operations (numpy, torch), data loading, visualization

### Documentation and Comments

- Document all functions with docstrings explaining purpose, parameters, and return values
- Add inline comments explaining the "why" behind non-obvious code
- Explain tensor dimension notation explicitly in comments

Example of dimension notation:
```python
# x: (B, C, H, W) - batch, channels, height, width
# B = batch size (number of samples processed together)
# C = channels (e.g., 3 for RGB images)
# H = height in pixels
# W = width in pixels
x = torch.randn(32, 3, 64, 64)

# latent: (B, D) - batch, latent dimension
# D = dimensionality of the latent space representation
latent = self.encoder(x)

# attn_weights: (B, N, S) - batch, num_queries, sequence_length
# N = number of query tokens
# S = length of key/value sequence
attn_weights = torch.softmax(scores, dim=-1)
```

### README Writing Style

Each folder's README should follow a first-principles reasoning structure:

1. **Start with the problem**: What are we trying to solve? Why is it hard?
2. **Pose questions**: What questions must we answer to solve this problem?
3. **Propose solutions**: Answer each question with a solution, explaining the intuition
4. **Show the code**: Link to relevant files and include code snippets with explanations

Structure template:
```markdown
# Component Name

[Brief intro establishing the problem]

## The Problem

[What we're trying to solve and why it's hard]

## The Questions

Out of first principles, we need to answer:
1. [Question 1]
2. [Question 2]
...

## The Solutions

### 1. [Answer to Question 1]

[Explanation with intuition]

From [path/to/file.py](path/to/file.py):

\`\`\`python
# Code snippet with comments
\`\`\`

[Explain the key insight]

### 2. [Answer to Question 2]
...
```

Guidelines:
- Use "what if" thinking to explore alternatives
- Link to source files: `From [models/fsq.py](models/fsq.py):`
- Include code snippets that show the key logic (not full implementations)
- Explain the intuition behind each design choice
- End with "What's next?" pointing to the next component

### Code Philosophy

This codebase prioritizes clarity and learning over optimization:
- Expect redundant code and minimal abstractions - this is intentional for building intuition
- Each folder should be self-contained and understandable in isolation
- Prefer explicit over implicit - show all the steps clearly
- When in doubt, write more comments explaining the intuition
