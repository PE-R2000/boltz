# Boltz-2 Input/Output Specifications

## Input Specifications

### 1. Sequence Inputs

#### Protein Sequences
```python
# Format: String of single-letter amino acid codes
sequence = "MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLI..."

# Tensor representation
res_type: torch.Tensor         # [batch, seq_len, 21] - one-hot amino acids
token_pad_mask: torch.Tensor   # [batch, seq_len] - valid sequence positions
```

#### DNA/RNA Sequences  
```python
# Format: String of nucleotide bases
sequence = "ATCGATCGATCG..."

# Tensor representation
res_type: torch.Tensor         # [batch, seq_len, 4] - one-hot nucleotides
```

#### Ligand Representations
```python
# SMILES string
smiles = "CC1=CC=CC=C1"  # Toluene example

# CCD code
ccd = "ATP"  # Adenosine triphosphate

# Tensor representation (after tokenization)
res_type: torch.Tensor         # [batch, seq_len, vocab_size] - tokenized representation
```

### 2. Multiple Sequence Alignment (MSA)

#### MSA Profile
```python
profile: torch.Tensor          # [batch, seq_len, 21] - amino acid frequencies
deletion_mean: torch.Tensor    # [batch, seq_len] - average deletion length
msa_mask: torch.Tensor         # [batch, msa_depth, seq_len] - valid MSA positions
```

#### MSA Input Formats
```python
# A3M format (standard)
msa_path = "protein.a3m"

# CSV format (with pairing keys)
# Columns: sequence, key
msa_df = pd.DataFrame({
    'sequence': ['MVTP...', 'MATP...', ...],
    'key': ['species1', 'species2', ...]
})
```

### 3. Atomic Structure Inputs

#### Coordinate Information
```python
coords: torch.Tensor           # [batch, conformers, atoms, 3] - xyz coordinates
atom_pad_mask: torch.Tensor    # [batch, atoms] - valid atom positions
atom_resolved_mask: torch.Tensor # [batch, atoms] - experimentally resolved atoms
```

#### Atomic Features
```python
atom_element: torch.Tensor     # [batch, atoms] - atomic numbers
atom_charge: torch.Tensor      # [batch, atoms] - formal charges
atom_is_aromatic: torch.Tensor # [batch, atoms] - aromaticity flags
atom_hybridization: torch.Tensor # [batch, atoms] - sp2/sp3 hybridization
```

### 4. Bond and Connectivity

#### Bond Graphs
```python
token_bonds: torch.Tensor      # [batch, seq_len, seq_len] - bond connectivity
bond_types: torch.Tensor       # [batch, seq_len, seq_len] - bond type (single/double/etc)
```

#### Covalent Bond Constraints
```yaml
# YAML specification
constraints:
  - bond:
      atom1: [chain_id, residue_idx, atom_name]  # ["A", 123, "SG"]
      atom2: [chain_id, residue_idx, atom_name]  # ["B", 45, "SG"]
```

### 5. Method Conditioning Inputs

#### Experimental Method
```python
method_feature: torch.Tensor   # [batch, seq_len] - method type encoding
# Method types:
# 0: X-ray crystallography
# 1: NMR spectroscopy
# 2: Cryo-electron microscopy  
# 3: Computational modeling
# 4: Unknown/other
```

#### Contact Conditioning
```python
contact_conditioning: torch.Tensor  # [batch, seq_len, seq_len, contact_types]
contact_threshold: torch.Tensor     # [batch, seq_len, seq_len] - distance thresholds

# Contact types:
# 0: UNSPECIFIED
# 1: UNSELECTED  
# 2: POCKET
# 3: INTERFACE
# 4: BINDING_SITE
# 5: ALLOSTERIC
```

### 6. Affinity-Specific Inputs

#### Ligand Identification
```python
affinity_token_mask: torch.Tensor  # [batch, seq_len] - ligand positions for affinity
mol_type: torch.Tensor             # [batch, seq_len] - molecule type (0=protein, 1=ligand)
affinity_mw: torch.Tensor          # [batch] - molecular weight for correction
```

#### Pairing Information
```yaml
# YAML specification for affinity
properties:
  - affinity:
      binder: "L"  # Chain ID of ligand
```

### 7. Template Inputs

#### Structural Templates
```python
template_coords: torch.Tensor      # [batch, templates, seq_len, atoms, 3]
template_mask: torch.Tensor        # [batch, templates, seq_len, atoms]
template_confidence: torch.Tensor  # [batch, templates, seq_len]
```

#### Template Specification
```yaml
# YAML specification
templates:
  - cif: "template.cif"
    chain_id: ["A", "B"]
    template_id: ["X", "Y"]  
    force: true
    threshold: 2.0  # Angstroms
```

## Output Specifications

### 1. Structure Predictions

#### 3D Coordinates
```python
sample_atom_coords: torch.Tensor   # [batch, samples, atoms, 3]
# - batch: Number of input complexes
# - samples: Number of diffusion samples (default: 1, can be up to 25)
# - atoms: Total number of atoms in complex
# - 3: x, y, z coordinates in Angstroms
```

#### Distance Distributions
```python
pdistogram: torch.Tensor          # [batch, seq_len, seq_len, distance_bins]
# Predicted probability distribution over distances for each residue pair
# Distance bins typically cover 2-22 Angstroms in 64 bins
```

### 2. Confidence Predictions

#### Per-Token Confidence
```python
plddt: torch.Tensor               # [batch, samples, seq_len] 
# Per-residue confidence score (0-100, higher = more confident)

pde: torch.Tensor                 # [batch, samples, seq_len]
# Per-residue distance error in Angstroms (lower = more confident)
```

#### Pairwise Confidence  
```python
pae: torch.Tensor                 # [batch, samples, seq_len, seq_len]
# Predicted Aligned Error between residue pairs in Angstroms

# Structure quality metrics
ptm: torch.Tensor                 # [batch, samples] - TM-score prediction (0-1)
iptm: torch.Tensor                # [batch, samples] - Interface TM-score (0-1)
```

#### Aggregated Confidence Metrics
```python
complex_plddt: torch.Tensor       # [batch, samples] - Average pLDDT across complex
complex_iplddt: torch.Tensor      # [batch, samples] - Interface-weighted pLDDT
complex_pde: torch.Tensor         # [batch, samples] - Average PDE across complex
complex_ipde: torch.Tensor        # [batch, samples] - Interface-weighted PDE

# Chain-specific metrics
ligand_iptm: torch.Tensor         # [batch, samples] - Protein-ligand interface TM
protein_iptm: torch.Tensor        # [batch, samples] - Protein-protein interface TM

# Overall quality score
confidence_score: torch.Tensor    # [batch, samples] 
# Calculated as: 0.8 * complex_plddt + 0.2 * (iptm if available else ptm)
```

### 3. Binding Affinity Predictions

#### Quantitative Affinity
```python
affinity_pred_value: torch.Tensor # [batch] - Predicted log(IC50) value
# Units: log(μM)
# Example interpretations:
#   -3: IC50 = 10^-9 M (strong binder)
#    0: IC50 = 10^-6 M (moderate binder) 
#    2: IC50 = 10^-4 M (weak binder/decoy)
```

#### Binary Classification
```python
affinity_probability_binary: torch.Tensor # [batch] - Binding probability (0-1)
# Probability that the ligand is a binder vs. decoy
# Use for hit discovery and binder classification
```

#### Ensemble Predictions (when enabled)
```python
affinity_pred_value1: torch.Tensor    # [batch] - First model prediction
affinity_pred_value2: torch.Tensor    # [batch] - Second model prediction
affinity_probability_binary1: torch.Tensor # [batch] - First model binary
affinity_probability_binary2: torch.Tensor # [batch] - Second model binary
```

### 4. Optional Outputs

#### B-Factor Predictions
```python
pbfactor: torch.Tensor            # [batch, seq_len, bfactor_bins] 
# Predicted B-factor distribution when enabled
```

#### Intermediate Representations
```python
s: torch.Tensor                   # [batch, seq_len, token_dim] - Final token representations
z: torch.Tensor                   # [batch, seq_len, seq_len, pair_dim] - Final pair representations
```

## Output File Formats

### 1. Structure Files

#### CIF/PDB Format
```
# Primary structure output
{input_name}_model_0.cif          # Best confidence structure
{input_name}_model_1.cif          # Second best structure
...
{input_name}_model_{N-1}.cif      # Nth structure (sorted by confidence)
```

#### Structure Content
- Atomic coordinates (ATOM records)
- Per-atom B-factors (derived from pLDDT)
- Chain identifiers
- Residue numbering
- Bond connectivity (in CIF format)

### 2. Confidence Files

#### JSON Format
```json
{
  "confidence_score": 0.8367,      // Overall quality metric (0-1)
  "ptm": 0.8425,                   // Predicted TM-score (0-1)
  "iptm": 0.8225,                  // Interface TM-score (0-1)
  "ligand_iptm": 0.7892,           // Protein-ligand interface TM (0-1)
  "protein_iptm": 0.8445,          // Protein-protein interface TM (0-1)
  "complex_plddt": 0.8402,         // Average pLDDT (0-1)
  "complex_iplddt": 0.8241,        // Interface-weighted pLDDT (0-1)
  "complex_pde": 0.8912,           // Average PDE (Angstroms)
  "complex_ipde": 5.1650,          // Interface PDE (Angstroms)
  "chains_ptm": {                  // Per-chain TM scores
    "0": 0.8533,                   // Chain A TM-score
    "1": 0.8330                    // Chain B TM-score
  },
  "pair_chains_iptm": {            // Inter-chain interface TM scores
    "0": {"0": 0.8533, "1": 0.8090},
    "1": {"0": 0.8225, "1": 0.8330}
  }
}
```

#### NumPy Arrays
```python
# Detailed per-token/pair arrays
pae_{input_name}_model_0.npz      # PAE matrix [seq_len, seq_len]
pde_{input_name}_model_0.npz      # PDE matrix [seq_len, seq_len]  
plddt_{input_name}_model_0.npz    # pLDDT array [seq_len]
```

### 3. Affinity Files

#### JSON Format
```json
{
  "affinity_pred_value": -1.23,           // Predicted log(IC50) in μM
  "affinity_probability_binary": 0.89,    // Binding probability (0-1)
  "affinity_pred_value1": -1.18,          // First model prediction
  "affinity_pred_value2": -1.28,          // Second model prediction  
  "affinity_probability_binary1": 0.87,   // First model binary
  "affinity_probability_binary2": 0.91    // Second model binary
}
```

## Input/Output Tensor Shapes Summary

### Input Tensor Dimensions
```python
# Sequence features
res_type: [B, N, vocab]           # B=batch, N=seq_len, vocab=21(protein)/4(DNA)
profile: [B, N, 21]               # MSA profile
deletion_mean: [B, N]             # MSA deletion statistics

# Structure features  
coords: [B, K, A, 3]              # K=conformers, A=atoms
atom_pad_mask: [B, A]             # Atom validity mask

# Conditioning features
method_feature: [B, N]            # Experimental method
contact_conditioning: [B, N, N, C] # C=contact_types
token_bonds: [B, N, N]            # Bond connectivity

# Affinity features
affinity_token_mask: [B, N]       # Ligand mask
mol_type: [B, N]                  # Molecule type
```

### Output Tensor Dimensions
```python
# Structure outputs
sample_atom_coords: [B, S, A, 3]  # S=diffusion_samples
pdistogram: [B, N, N, D]          # D=distance_bins

# Confidence outputs
plddt: [B, S, N]                  # Per-token confidence
pae: [B, S, N, N]                 # Pairwise error
ptm: [B, S]                       # Structure quality
iptm: [B, S]                      # Interface quality

# Affinity outputs  
affinity_pred_value: [B]          # Binding affinity
affinity_probability_binary: [B]  # Binding probability

# Representations
s: [B, N, token_dim]              # Token embeddings (384-768)
z: [B, N, N, pair_dim]            # Pair embeddings (128-256)
```

This comprehensive input/output specification provides the detailed information needed to understand exactly what data Boltz-2 expects and produces, enabling effective integration into computational pipelines and proper interpretation of results.