# MoE-CAP

## Requirements
Python: >= 3.9

## Installation
```bash
git clone https://github.com/SecretSettler/MoE-CAP.git
cd MoE-CAP
pip install -e .
```
Then you can import `moe_cap` directly.

## Quick Example
1. Launch our sglang custom server (e.g. H100)
```bash
python -m moe_cap.systems.sglang \ 
        --model-path Qwen/Qwen3-235B-A22B-Thinking-2507 \
        --port 30000 \
        --expert-distribution-recorder-mode stat \
        --tp-size 8
```

2. Run our benchmark
```bash
python -m moe_cap.runner.sglang_profile \
        --config-file configs/gsm8k_qwen3_235b_a22b.yaml \
        --output_dir outputs/
```

## Cite our paper
```
@misc{jiang2025moecapbenchmarkingcostaccuracy,
      title={MoE-CAP: Benchmarking Cost, Accuracy and Performance of Sparse Mixture-of-Experts Systems}, 
      author={Yinsicheng Jiang and Yao Fu and Yeqi Huang and Ping Nie and Zhan Lu and Leyang Xue and Congjie He and Man-Kit Sit and Jilong Xue and Li Dong and Ziming Miao and Dayou Du and Tairan Xu and Kai Zou and Edoardo Ponti and Luo Mai},
      year={2025},
      eprint={2412.07067},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.07067}, 
}
```