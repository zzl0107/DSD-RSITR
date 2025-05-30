# DSD-RSITR
## 📊 Datasets

This project is based on two remote sensing datasets:

- [RSICD](https://github.com/201528014227051/RSICD_optimal)  
- [RSITMD](https://github.com/xiaoyuan1996/AMFMN/tree/master/RSITMD)

The image data can be downloaded from [Baidu Disk](https://pan.baidu.com/s/1jRO56hYy1CtciDycJt_uJQ?pwd=icnr).  
After downloading, please update the `image_root` field in the following config files with the local path to your image folder:

- `configs/Retrieval_rsicd_vit.yaml`  
- `configs/Retrieval_rsitmd_vit.yaml`

Text has already been pre-processed in [PIR](https://github.com/jaychempan/PIR) and stored under the `data/finetune` directory. These are referenced using relative paths in the config files and do not require modification.

---

## 🚀 Training

The original [PIR](https://github.com/jaychempan/PIR) framework uses distributed training.  
We provide a **single-GPU training mode** (requires a GPU with at least **24GB memory**) by modifying `run.py`.

Please set your target GPU device in `run.py`, for example:

```python
args.device = "cuda:0"
```

Then, use the following commands to start training:

```bash
# For RSITMD
python run.py --task 'itr_rsitmd_vit' --dist "single" \
              --config 'configs/Retrieval_rsitmd_vit.yaml' \
              --output_dir './checkpoints/full_rsitmd_vit'

# For RSICD
python run.py --task 'itr_rsicd_vit' --dist "single" \
              --config 'configs/Retrieval_rsicd_vit.yaml' \
              --output_dir './checkpoints/full_rsicd_vit'
```

The pretrained CLIP model is loaded via [OpenCLIP](https://github.com/mlfoundations/open_clip).  
If the weights fail to load automatically, you can download them manually from [Hugging Face](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K), and **modify the local path** in `models/model_retrieval.py` accordingly.

---

## 🧪 Evaluation

To evaluate a trained model, use the following commands:

```bash
# RSITMD evaluation
python run.py --task 'itr_rsitmd_vit' --dist "single" \
              --config 'configs/Retrieval_rsitmd_vit.yaml' \
              --output_dir './checkpoints/test' \
              --checkpoint './checkpoints/full_rsitmd_vit/checkpoint_best.pth' \
              --evaluate

# RSICD evaluation
python run.py --task 'itr_rsicd_vit' --dist "single" \
              --config 'configs/Retrieval_rsicd_vit.yaml' \
              --output_dir './checkpoints/test' \
              --checkpoint './checkpoints/full_rsicd_vit/checkpoint_best.pth' \
              --evaluate
```

---

## 🙏 Acknowledgements

This project is built upon the excellent work of:

- [PIR](https://github.com/jaychempan/PIR)  
- [HarMA](https://github.com/seekerhuang/HarMA)
