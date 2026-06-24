# Indic Video Dataset Pipeline

Code only under `Indic_video_pipeline/`. All runtime outputs go to **`pipeline_outputs/`**. Models live in **`models/`**.

Orchestrated by `run_pipeline.py` — 12 services (s1–s12) share state via **`metadata.jsonl`** per movie workspace.

## Pipeline

```mermaid
flowchart TB

    %% Input
    A["🎬 Movie MP4"]

    %% Preprocessing
    subgraph P["Preprocessing (CPU)"]
        B["s1 Extract Clips<br/>PySceneDetect · 5s clips"]
        C["s2 Dedup / Filter<br/>UniMatch Motion · DOVER"]
        D["s3 Band Removal"]
        E["s4 Watermark Detection"]
    end

    %% Understanding
    subgraph U["Understanding (GPU)"]
        F["s5 Classification<br/>Qwen2.5-VL · 12 Buckets"]
        G["s6 Verification<br/>Gemma"]
    end

    %% Enrichment
    subgraph EN["Enrichment (GPU)"]
        H["s7 Actor Tagging<br/>YOLO Face + InsightFace"]
        I["s8 Caption Generation<br/>Qwen2.5-VL Video"]
    end

    %% Quality
    subgraph Q["Quality & Ranking"]
        J["s9 Quality Score<br/>CLIP + DOVER + Motion"]
        K["s10 Gate<br/>DISCARD · REVIEW · FINAL"]
    end

    %% Export
    subgraph EX["Export"]
        L["s11 Export<br/>JSONL · CSV · Clips"]
        M["s12 Report"]
    end

    %% Metadata Store
    META[("metadata.jsonl")]

    %% Actor Database
    ACTOR[("Actor Embeddings<br/>Database")]

    %% Main Pipeline
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G

    G -->|"People Bucket"| H
    G -->|"Other Buckets"| I

    H --> I
    I --> J
    J --> K
    K --> L
    L --> M

    %% Metadata Updates
    B -.-> META
    C -.-> META
    D -.-> META
    E -.-> META
    F -.-> META
    G -.-> META
    H -.-> META
    I -.-> META
    J -.-> META
    K -.-> META
    L -.-> META

    %% External Actor DB
    ACTOR -.-> H

    %% Styling
    classDef input fill:#dbeafe,stroke:#2563eb,stroke-width:2px;
    classDef prep fill:#dcfce7,stroke:#16a34a;
    classDef understand fill:#fef3c7,stroke:#d97706;
    classDef enrich fill:#ede9fe,stroke:#7c3aed;
    classDef quality fill:#fee2e2,stroke:#dc2626;
    classDef export fill:#e0f2fe,stroke:#0891b2;

    class A input;
    class B,C,D,E prep;
    class F,G understand;
    class H,I enrich;
    class J,K quality;
    class L,M export;
```

## Layout

```text
Indic_video_pipeline/          # code, configs, services (no logs/jsonl here)
/mnt/data0/harsha/new_dataset_pipeline/
  models/                      # Qwen2.5-VL, yolov12n-face.pt
  pipeline_outputs/
    workspaces/<video_id>/     # metadata.jsonl, export/, actor_frames/
    logs/s1..s12/
    reports/
  master/                      # actor_tagger, captioner, actor_embeddings/
```

## Setup

```bash
conda activate indic_video_pipeline
cd /mnt/data0/harsha/new_dataset_pipeline/Indic_video_pipeline
bash scripts/clean_generated.sh      # reset outputs
bash scripts/setup_all_models.sh     # Qwen + YOLO → models/
bash scripts/fix_cuda_torch.sh       # if CUDA false
```

## Run — Devdas

```bash
python run_pipeline.py \
  --movie /mnt/data0/parth/world_models/HunyuanVideo-Avatar/assets/devdas_standard.mp4 \
  --video-id devdas_standard \
  --force
```

Actor tagging only (s1–s7):

```bash
python run_pipeline.py \
  --movie /mnt/data0/parth/world_models/HunyuanVideo-Avatar/assets/devdas_standard.mp4 \
  --video-id devdas_standard \
  --from-step s1 \
  --to-step s7 \
  --force
```

## Outputs

| Type | Path |
|------|------|
| metadata.jsonl | `pipeline_outputs/workspaces/devdas_standard/metadata.jsonl` |
| export csv/jsonl | `pipeline_outputs/workspaces/devdas_standard/export/` |
| actor tags | `pipeline_outputs/workspaces/devdas_standard/actor_tags/` |
| runtime logs | `pipeline_outputs/logs/s*/` |
| reports | `pipeline_outputs/reports/` |

Actor embeddings: `master/actors/actor_embeddings/` (108 actors).
