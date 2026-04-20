# LLaVE / CV-Deepseed — Project README

This repository contains the LLaVE-based multimodal embedding code and experiments used to develop language-and-vision embedding models with hardness-weighted contrastive learning. This README summarizes key files, fixes, and quick usage for evaluation and training using the repository as currently arranged.

**Summary**
- **Purpose**: Train and evaluate unified multimodal embedding models (image/text/video) with improved retrieval quality and language-aware hardness weighting.
- **Reference README**: See the original notes in [OldREADME.md](OldREADME.md).

**Key Files (edited / relevant)**
- **Evaluation script**: [evaluate_retrieval.py](evaluate_retrieval.py) — main retrieval/evaluation entrypoint (robust JSON text extraction, float32 similarity calculations).
- **Model builder**: [llava/model/builder.py](llava/model/builder.py) — modified to avoid device_map no-op weight loading on Windows and set low_cpu_mem_usage appropriately.
- **Model architecture**: [llava/model/language_model/llava_qwen.py](llava/model/language_model/llava_qwen.py) — added NoneType attention_mask checks and dynamic pooling_type selection.
- **Training entry & trainer**: [llava/train/train.py](llava/train/train.py) and [llava/train/llava_trainer.py](llava/train/llava_trainer.py) — update ModelArguments defaults and contain the language-aware hardness weighting compute_loss.
- **Linguistic similarity util**: [Flickr30k/simularity.py](Flickr30k/simularity.py)
- **Dataset JSON example**: [Flickr30k/flickr_llave.json](Flickr30k/flickr_llave.json)

**Quick Setup**
- **Environment**: create a conda env with Python 3.10 (matching training env) and install requirements in `requirements.txt`.

Example (Windows/PowerShell):
```powershell
conda create -n llave_stable python=3.10 -y
conda activate llave_stable
pip install -r requirements.txt
pip install -e .
```

**Training example**
```bash
C:\Users\himsi\miniconda3\envs\llave_stable\python.exe llava/train/train_mem.py \
  --model_name_or_path zhibinlan/LLaVE-2B \
  --data_path ./Flickr30k/flickr_llave.json \
  --image_folder ./Flickr30k/ \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --alpha 9 \
  --beta 1 \
  --ling_emb_path ./Flickr30k/ling_embeddings.pt \
  --output_dir ./checkpoints/language_aware_beta2 \
  --bf16 True \
  --tf32 True \
  --attn_implementation sdpa \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --dataloader_num_workers 8 \
  --learning_rate 2e-4 \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --lazy_preprocess True \
  --group_by_modality_length True \
  --report_to none --logging_steps 1
```

**Evaluation example**
```bash
python evaluate_retrieval.py --model_path ./checkpoints/language_aware_base \
  --data_path ./Flickr30k/flickr_llave.json --image_folder ./Flickr30k/flickr30k-images/
```

**Notes & Recommendations**
- **Windows weight loading**: Changes in [llava/model/builder.py](llava/model/builder.py) disable passing `device_map` in kwargs and set `low_cpu_mem_usage=False` to avoid a no-op weights load on Windows.
- **Retrieval fixes**: `evaluate_retrieval.py` was updated to use float32 similarity compute and more robust text extraction to fix blurry/low-quality retrieval scores.

**Acknowledgements**
- Derived and adapted from the original LLaVE / LLaVA-NeXT work and datasets referenced in [OldREADME.md](OldREADME.md).

**Citation**
See the original citation block in [OldREADME.md](OldREADME.md).
