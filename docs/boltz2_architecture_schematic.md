# Boltz-2 Architecture Schematic

## Visual Model Architecture

```
                                    BOLTZ-2 MODEL ARCHITECTURE
                                    ═══════════════════════════

INPUT LAYER                        EMBEDDING LAYER                     TRUNK PROCESSING
═══════════                        ══════════════                      ════════════════

┌─────────────────┐               ┌─────────────────┐                 ┌─────────────────┐
│   Sequences     │               │                 │                 │                 │
│  ┌───────────┐  │               │  InputEmbedder  │                 │   MSA Module    │
│  │ Proteins  │  │──────────────▶│                 │────────────────▶│                 │
│  │ DNA/RNA   │  │               │ ┌─────────────┐ │                 │ ┌─────────────┐ │
│  │ Ligands   │  │               │ │AtomEncoder  │ │                 │ │MSA Profile  │ │
│  └───────────┘  │               │ │Token Embed  │ │                 │ │Processing   │ │
├─────────────────┤               │ │Residue Type │ │                 │ │Outer Product│ │
│      MSA        │               │ └─────────────┘ │                 │ └─────────────┘ │
│  ┌───────────┐  │               └─────────────────┘                 └─────────────────┘
│  │ Profiles  │  │                        │                                   │
│  │ Evolution │  │                        ▼                                   ▼
│  │ Info      │  │               ┌─────────────────┐                 ┌─────────────────┐
│  └───────────┘  │               │Method Condition │                 │  Pairformer     │
├─────────────────┤               │Contact Condition│                 │     Stack       │
│  Constraints    │               │Bond Features    │                 │                 │
│  ┌───────────┐  │               │Positional Enc  │                 │ ┌─────────────┐ │
│  │ Contacts  │  │──────────────▶│                 │────────────────▶│ │Row/Col Attn │ │
│  │ Pockets   │  │               │                 │                 │ │Transition   │ │
│  │ Bonds     │  │               │ s: [B,N,Ds]     │                 │ │Layer Norm   │ │
│  └───────────┘  │               │ z: [B,N,N,Dz]   │                 │ │×num_layers  │ │
├─────────────────┤               └─────────────────┘                 │ └─────────────┘ │
│   Templates     │                                                   └─────────────────┘
│  ┌───────────┐  │                                                            │
│  │ Struct    │  │                        RECYCLING LOOP                     │
│  │ Info      │  │               ┌─────────────────────────────────────────────▼─────┐
│  └───────────┘  │               │           (Repeated R times)                      │
├─────────────────┤               │  ┌───────────┐  ┌────────────┐  ┌─────────────┐  │
│   Method Info   │               │  │Template   │  │MSA Module  │  │ Pairformer  │  │
│  ┌───────────┐  │               │  │Module     │  │            │  │   Module    │  │
│  │ X-ray     │  │               │  │(optional) │  │            │  │             │  │
│  │ NMR       │  │               │  └───────────┘  └────────────┘  └─────────────┘  │
│  │ Cryo-EM   │  │               └───────────────────────────────────────────────────┘
│  └───────────┘  │                                        │
└─────────────────┘                                        ▼
                                                  ┌─────────────────┐
                                                  │Final s,z Reprs  │
                                                  │s: [B,N,Ds]      │
                                                  │z: [B,N,N,Dz]    │
                                                  └─────────────────┘
                                                           │
                                                           ▼

STRUCTURE GENERATION               CONDITIONING                        OUTPUT MODULES
═══════════════════               ═══════════                        ══════════════

┌─────────────────┐               ┌─────────────────┐                ┌─────────────────┐
│Diffusion        │               │Diffusion        │                │   Structure     │
│Conditioning     │               │Process          │                │                 │
│                 │               │                 │                │ ┌─────────────┐ │
│┌─────────────┐  │               │┌─────────────┐  │                │ │3D Coords    │ │
││Pairwise     │  │──────────────▶││Noise        │  │───────────────▶│ │[B,S,A,3]    │ │
││Conditioning │  │               ││Scheduling   │  │                │ └─────────────┘ │
│└─────────────┘  │               │└─────────────┘  │                └─────────────────┘
│┌─────────────┐  │               │┌─────────────┐  │                ┌─────────────────┐
││Atom Encoder │  │               ││Denoising    │  │                │  Confidence     │
││             │  │               ││Transformer  │  │                │                 │
│└─────────────┘  │               │└─────────────┘  │                │ ┌─────────────┐ │
│┌─────────────┐  │               │┌─────────────┐  │                │ │pLDDT        │ │
││Query/Key    │  │               ││Score        │  │                │ │PAE          │ │
││Generation   │  │               ││Network      │  │                │ │PTM/iPTM     │ │
│└─────────────┘  │               │└─────────────┘  │                │ └─────────────┘ │
└─────────────────┘               └─────────────────┘                └─────────────────┘
         │                                 │                         ┌─────────────────┐
         ▼                                 ▼                         │   Distogram     │
┌─────────────────┐               ┌─────────────────┐                │                 │
│Attention Biases │               │Iterative       │                │ ┌─────────────┐ │
│                 │               │Refinement      │                │ │Distance     │ │
│┌─────────────┐  │               │                │                │ │Distributions│ │
││Atom Encoder │  │               │T steps of:     │                │ │[B,N,N,Bins] │ │
││Bias         │  │               │1. Add noise    │                │ └─────────────┘ │
│└─────────────┘  │               │2. Predict     │                │                 │
│┌─────────────┐  │               │3. Denoise     │                └─────────────────┘
││Token Trans  │  │               │4. Update      │                ┌─────────────────┐
││Bias         │  │               │               │                │    Affinity     │
│└─────────────┘  │               └─────────────────┘                │   (Boltz-2)     │
│┌─────────────┐  │                        │                        │                 │
││Atom Decoder │  │                        ▼                        │ ┌─────────────┐ │
││Bias         │  │               ┌─────────────────┐                │ │Binding      │ │
│└─────────────┘  │               │Final 3D         │                │ │Affinity     │ │
└─────────────────┘               │Coordinates      │                │ │log(IC50)    │ │
                                  │[B,S,A,3]        │                │ └─────────────┘ │
                                  └─────────────────┘                │ ┌─────────────┐ │
                                                                     │ │Binary       │ │
                                                                     │ │Probability  │ │
                                                                     │ │[0,1]        │ │
                                                                     │ └─────────────┘ │
                                                                     └─────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════
                                    KEY DIMENSIONS
═══════════════════════════════════════════════════════════════════════════════════════

Input Dimensions:                  Processing Dimensions:              Output Dimensions:
B = Batch Size                      Ds = Token embedding (384-768)      S = Diffusion samples (1-25)
N = Sequence Length                 Dz = Pair embedding (128-256)       A = Total atoms
A = Total Atoms                     R = Recycling steps (1-10)          C = Confidence metrics
C = Contact Types                   T = Diffusion steps (200)           
```

## Conditioning Flow Diagram

```
                              METHOD CONDITIONING EFFECTS
                              ══════════════════════════

Input Method Type                    Processing Impact                   Output Effects
═════════════════                    ════════════════                   ══════════════

┌─────────────┐                     ┌─────────────────┐                ┌─────────────────┐
│             │                     │                 │                │                 │
│   X-ray     │────────────────────▶│ High Precision  │───────────────▶│ pLDDT: 80-95    │
│             │                     │ Expectations    │                │ PAE: 1-3 Å      │
└─────────────┘                     └─────────────────┘                └─────────────────┘

┌─────────────┐                     ┌─────────────────┐                ┌─────────────────┐
│             │                     │                 │                │                 │
│    NMR      │────────────────────▶│ Dynamic Info    │───────────────▶│ pLDDT: 70-85    │
│             │                     │ Flexibility     │                │ Ensemble Aware  │
└─────────────┘                     └─────────────────┘                └─────────────────┘

┌─────────────┐                     ┌─────────────────┐                ┌─────────────────┐
│             │                     │                 │                │                 │
│  Cryo-EM    │────────────────────▶│ Resolution      │───────────────▶│ Variable Quality │
│             │                     │ Dependent       │                │ Large Complexes │
└─────────────┘                     └─────────────────┘                └─────────────────┘

                              CONTACT CONDITIONING EFFECTS
                              ═══════════════════════════

Contact Type                        Conditioning Signal                 Structural Impact
════════════                        ═══════════════════                 ═════════════════

┌─────────────┐                     ┌─────────────────┐                ┌─────────────────┐
│             │                     │                 │                │                 │
│   POCKET    │────────────────────▶│ Distance        │───────────────▶│ Improved Ligand │
│             │                     │ Constraints     │                │ Placement       │
└─────────────┘                     └─────────────────┘                └─────────────────┘

┌─────────────┐                     ┌─────────────────┐                ┌─────────────────┐
│             │                     │                 │                │                 │
│ INTERFACE   │────────────────────▶│ Contact         │───────────────▶│ Better PPI      │
│             │                     │ Patterns        │                │ Prediction      │
└─────────────┘                     └─────────────────┘                └─────────────────┘

┌─────────────┐                     ┌─────────────────┐                ┌─────────────────┐
│             │                     │                 │                │                 │
│BINDING_SITE │────────────────────▶│ Focused         │───────────────▶│ Enhanced        │
│             │                     │ Attention       │                │ Affinity Pred   │
└─────────────┘                     └─────────────────┘                └─────────────────┘

```

## Affinity Prediction Pipeline

```
                                AFFINITY PREDICTION FLOW
                                ═══════════════════════

Input Complex                       Processing Steps                    Affinity Outputs
═════════════                       ════════════════                    ════════════════

┌─────────────┐                    ┌─────────────────┐                 ┌─────────────────┐
│Protein +    │                    │1. Structure     │                 │                 │
│Ligand       │───────────────────▶│   Generation    │                 │ affinity_pred_  │
│Pair         │                    │                 │                 │ value           │
│             │                    └─────────────────┘                 │ (log IC50)      │
└─────────────┘                             │                          │                 │
                                            ▼                          └─────────────────┘
┌─────────────┐                    ┌─────────────────┐                 ┌─────────────────┐
│Cross-       │                    │2. Best Structure│                 │                 │
│Interaction  │◀───────────────────│   Selection     │                 │ affinity_prob_  │
│Masking      │                    │   (by iPTM)     │                 │ binary          │
│             │                    └─────────────────┘                 │ (0-1)           │
└─────────────┘                             │                          │                 │
      │                                     ▼                          └─────────────────┘
      ▼                            ┌─────────────────┐                 ┌─────────────────┐
┌─────────────┐                    │3. Affinity      │                 │                 │
│Ligand-      │                    │   Specific      │                 │ Ensemble        │
│Receptor     │───────────────────▶│   Embedding     │────────────────▶│ Predictions     │
│Focus        │                    │                 │                 │ (if enabled)    │
│             │                    └─────────────────┘                 │                 │
└─────────────┘                             │                          └─────────────────┘
                                            ▼
                                   ┌─────────────────┐
                                   │4. Affinity      │
                                   │   Transformer   │
                                   │   Processing    │
                                   └─────────────────┘
```

This visual schematic provides a comprehensive overview of the Boltz-2 architecture, showing the flow from inputs through processing to outputs, with special emphasis on the conditioning mechanisms that make Boltz-2 unique in its ability to predict both structure and binding affinity.