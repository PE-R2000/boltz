# Boltz-2 Model Architecture: Deep Analysis

## Executive Summary

Boltz-2 is a biomolecular foundation model that jointly predicts complex structures and binding affinities. This document provides a comprehensive analysis of the model architecture, input/output specifications, and the effects of method conditioning on model outputs.

## Table of Contents

1. [Overall Architecture Overview](#overall-architecture-overview)
2. [Input Processing Pipeline](#input-processing-pipeline)
3. [Core Model Components](#core-model-components)
4. [Output Generation](#output-generation)
5. [Method Conditioning Effects](#method-conditioning-effects)
6. [Detailed Component Analysis](#detailed-component-analysis)
7. [Data Flow Diagram](#data-flow-diagram)

## Overall Architecture Overview

Boltz-2 follows a multi-stage architecture that processes biomolecular inputs through several key stages:

```
Input Features → Input Embedding → Trunk Processing → Structure Generation → Output Predictions
     ↓               ↓                   ↓                    ↓                  ↓
  Sequences,      Atom & Token       MSA Module,         Diffusion         3D Coordinates,
  MSA, Bonds,     Embeddings,        Pairformer,         Sampling          Confidence,
  Contacts,       Positional         Template            Process           Affinity
  Method Info     Encoding           Processing
```

### Key Innovation: Joint Structure-Affinity Prediction

Unlike previous models that focus solely on structure prediction, Boltz-2 simultaneously predicts:
- **3D atomic coordinates** (structure)
- **Binding affinity values** (`log(IC50)`)
- **Binary binding probability** (binder vs. decoy classification)
- **Confidence scores** (pLDDT, PAE, PTM, iPTM)

## Input Processing Pipeline

### 1. Input Types and Formats

**Sequence Information:**
- Protein sequences (amino acids)
- DNA/RNA sequences (nucleotides) 
- Ligand representations (SMILES strings or CCD codes)
- Multiple Sequence Alignments (MSA)

**Structural Information:**
- Atomic coordinates (for training/templates)
- Bond connectivity graphs
- Covalent bond constraints
- Distance constraints

**Method Conditioning:**
- Experimental method types (X-ray, NMR, Cryo-EM, etc.)
- Resolution information
- Modified residue specifications

### 2. Feature Extraction and Tokenization

The `InputEmbedder` processes raw inputs into structured features:

```python
# Key input features extracted:
feats = {
    'res_type': one_hot_amino_acids,           # [B, N, 21] 
    'profile': msa_profile,                    # [B, N, 21]
    'deletion_mean': msa_deletions,            # [B, N]
    'atom_positions': atom_coords,             # [B, A, 3]
    'atom_mask': atom_padding,                 # [B, A]
    'token_bonds': bond_connectivity,          # [B, N, N]
    'method_feature': experimental_method,     # [B, N]
    'contact_conditioning': distance_constraints, # [B, N, N, C]
    'affinity_token_mask': ligand_mask,       # [B, N] (for affinity prediction)
}
```

## Core Model Components

### 1. Input Embedder (`InputEmbedder`)

**Purpose:** Converts raw molecular features into learned embeddings

**Architecture:**
- Atom-level encoder with attention mechanisms
- Token-level residue type and MSA profile encodings
- Method conditioning integration
- Molecular type differentiation (protein, DNA, RNA, ligand)

**Key Operations:**
```python
s = atom_embedding + residue_encoding + msa_encoding + method_conditioning
```

### 2. Trunk Processing Stack

#### MSA Module (`MSAModule`)
- Processes multiple sequence alignments
- Computes evolutionary information
- Updates pairwise representations with MSA features

#### Pairformer Module (`PairformerModule`)
- Core transformer architecture for pairwise reasoning
- Attention between sequence positions
- Updates both single (s) and pairwise (z) representations

#### Template Module (`TemplateModule/TemplateV2Module`)
- Incorporates structural templates when available
- Processes template coordinates and alignments
- Projects template information into pairwise representations

### 3. Structure Generation (`AtomDiffusion`)

**Diffusion Process:**
- Uses a denoising diffusion probabilistic model
- Generates 3D atomic coordinates iteratively
- Conditioned on trunk representations and method information

**Key Components:**
- `DiffusionConditioning`: Prepares conditioning signals
- `DiffusionTransformer`: Core diffusion transformer
- Noise scheduling and sampling procedures

### 4. Confidence Prediction (`ConfidenceModule`)

**Predicted Metrics:**
- **pLDDT:** Per-residue confidence (0-100)
- **PAE:** Predicted Aligned Error between residue pairs
- **PTM:** Predicted TM-score for overall structure quality
- **iPTM:** Interface PTM for inter-chain interactions

### 5. Affinity Prediction (`AffinityModule`)

**Novel Component in Boltz-2:**
- Predicts binding affinity as `log(IC50)` in μM
- Binary classification for binder vs. decoy
- Ensemble prediction for improved accuracy
- Molecular weight correction option

## Output Generation

### Structure Outputs

**Primary Structure:**
- `sample_atom_coords`: 3D coordinates `[B, S, A, 3]`
  - B: batch size
  - S: diffusion samples  
  - A: number of atoms
  - 3: x, y, z coordinates

**Distance Information:**
- `pdistogram`: Distance distribution predictions `[B, N, N, bins]`

### Confidence Outputs

**Token-level Confidence:**
- `plddt`: Per-token confidence scores `[B, S, N]`
- `pde`: Per-token distance error `[B, S, N]`

**Pairwise Confidence:**
- `pae`: Predicted aligned error `[B, S, N, N]`
- `ptm`: TM-score prediction `[B, S]`
- `iptm`: Interface TM-score `[B, S]`

**Aggregated Metrics:**
- `confidence_score`: Overall quality score (0.8 * pLDDT + 0.2 * iPTM)
- `complex_plddt`: Average pLDDT across complex
- `ligand_iptm`: Interface quality for protein-ligand interactions

### Affinity Outputs

**Quantitative Affinity:**
- `affinity_pred_value`: Predicted `log(IC50)` value
  - IC50 of 10^-9 M → model outputs -3 (strong binder)
  - IC50 of 10^-6 M → model outputs 0 (moderate binder)  
  - IC50 of 10^-4 M → model outputs 2 (weak binder)

**Binary Classification:**
- `affinity_probability_binary`: Probability ligand is a binder (0-1)

**Ensemble Predictions:**
- Individual model predictions (`affinity_pred_value1`, `affinity_pred_value2`)
- Averaged ensemble prediction

## Method Conditioning Effects

### 1. Experimental Method Conditioning

The model incorporates experimental method information that affects predictions:

**Method Types:**
- X-ray crystallography
- NMR spectroscopy
- Cryo-electron microscopy
- Computational modeling

**Implementation:**
```python
if self.add_method_conditioning:
    s = s + self.method_conditioning_init(feats["method_feature"])
```

**Effects on Outputs:**
- **Structure Quality:** Different methods have different accuracy expectations
- **Confidence Calibration:** Method-specific confidence scoring
- **Resolution Dependencies:** High-resolution methods get higher confidence

### 2. Contact Conditioning

**Distance Constraints:**
- Pocket conditioning for ligand binding sites
- Inter-chain contact specifications
- Experimental distance constraints

**Implementation:**
```python
contact_conditioning = self.contact_conditioning(feats)
z_init = z_init + contact_conditioning
```

**Effects:**
- **Guided Structure Generation:** Enforces known distance constraints
- **Improved Binding Poses:** Better ligand placement in known pockets
- **Enhanced Interface Prediction:** More accurate protein-protein interfaces

### 3. Diffusion Conditioning

**Multi-level Conditioning:**
- Token-level conditioning from trunk representations
- Atom-level conditioning for fine-grained control
- Pairwise conditioning for interaction modeling

**Components:**
```python
diffusion_conditioning = {
    'q': query_representations,          # [B, A, D]
    'c': context_representations,        # [B, A, D] 
    'to_keys': key_projections,         # [B, A, D]
    'atom_enc_bias': encoder_bias,      # [B, A, H*L]
    'atom_dec_bias': decoder_bias,      # [B, A, H*L]
    'token_trans_bias': transformer_bias # [B, N, N, H*L]
}
```

**Effects on Structure Generation:**
- **Iterative Refinement:** Gradual improvement through diffusion steps
- **Multi-scale Modeling:** Atom and residue level information integration
- **Context-aware Sampling:** Conditioning ensures consistency with input features

### 4. Affinity-Specific Conditioning

**Ligand-Focused Processing:**
- Special handling for affinity prediction
- Cross-interaction masking between ligand and target
- Best structure selection for affinity calculation

**Implementation:**
```python
# Focus on ligand-target interactions
cross_pair_mask = (lig_mask[:, None] * rec_mask[None, :] + 
                   rec_mask[:, None] * lig_mask[None, :] + 
                   lig_mask[:, None] * lig_mask[None, :])
z_affinity = z * cross_pair_mask[None, :, :, None]
```

## Detailed Component Analysis

### Input Feature Dimensions

**Sequence Features:**
- `token_s`: Token embedding dimension (typically 384-768)
- `token_z`: Pairwise embedding dimension (typically 128-256)
- `atom_s`: Atom embedding dimension (typically 128)
- `atom_z`: Atom pairwise embedding dimension (typically 64)

**Attention Windows:**
- `atoms_per_window_queries`: 32 (memory efficiency)
- `atoms_per_window_keys`: 128 (context window)

### Model Scaling

**Memory Management:**
- Gradient checkpointing for large models
- Windowed attention for atom-level processing
- Optional compilation for acceleration

**Training Efficiency:**
- EMA (Exponential Moving Average) for stable training
- Mixed precision training support
- Multiple recycling steps during training

### Loss Functions

**Structure Prediction:**
- Diffusion loss for coordinate generation
- Distogram loss for distance prediction
- B-factor prediction loss (optional)

**Confidence Prediction:**
- pLDDT loss against true confidence
- PAE loss for error prediction
- PTM loss for structure quality

**Affinity Prediction:**
- Regression loss for binding affinity values
- Binary classification loss for binder detection

## Data Flow Diagram

```
Input Sequences/Structures
          ↓
    InputEmbedder
          ↓
   [s_inputs: token embeddings]
          ↓
    Initialize s, z representations
          ↓
    ┌─────────────────────────┐
    │   Recycling Loop        │
    │  ┌─────────────────┐    │
    │  │   MSA Module    │────┼─→ z updates
    │  └─────────────────┘    │
    │  ┌─────────────────┐    │
    │  │  Pairformer     │────┼─→ s, z updates  
    │  └─────────────────┘    │
    │  ┌─────────────────┐    │
    │  │ Template Module │────┼─→ z updates (optional)
    │  └─────────────────┘    │
    └─────────────────────────┘
          ↓
    Final s, z representations
          ↓
    ┌─────────────────────────┐
    │ Diffusion Conditioning  │
    └─────────────────────────┘
          ↓
    ┌─────────────────────────┐
    │   Structure Module      │
    │   (AtomDiffusion)       │
    └─────────────────────────┘
          ↓
    3D Coordinates
          ↓
    ┌───────────────┬─────────────────┐
    │               │                 │
    ↓               ↓                 ↓
Confidence     Affinity       Distogram
Module         Module         Module
    ↓               ↓                 ↓
pLDDT, PAE,    Binding           Distance
PTM, iPTM      Affinity          Distributions
```

## Conclusion

Boltz-2 represents a significant advancement in biomolecular modeling by:

1. **Joint Prediction:** Simultaneously predicting structure and binding affinity
2. **Method Awareness:** Incorporating experimental method conditioning
3. **Flexible Input Handling:** Supporting diverse molecular types and constraints
4. **Comprehensive Outputs:** Providing structure, confidence, and affinity predictions
5. **Scalable Architecture:** Efficient memory usage and computational scaling

The method conditioning system allows the model to adapt its predictions based on experimental context, leading to more accurate and contextually appropriate outputs for different use cases in drug discovery and structural biology.