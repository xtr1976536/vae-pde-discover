# VAE–PDE Discover

Twenty-frame forecasts from twenty observed velocity frames on the RealPDEBench Cylinder real-data split. All runs share 8,141 training, 1,234 validation, and 4,827 test windows. The split indexes and no-overlap audit are in [data/splits](data/splits). The run manifest records the experiment code, source run, and snapshot timestamp. Each result directory contains the validation history and any completed test metrics. See [results](results) and [manifest.json](manifest.json).

## Results snapshot

Snapshot time 2026-11-25T14:22:13.625814+08:00.

| Run | Status | Completed epochs | Best validation RMSE | Test RMSE |
|---|---|---:|---:|---:|
| vpc_joint_c8 | RUNNING | 10 | 0.20912 | Pending |
| vpc_joint_c16 | RUNNING | 10 | 0.20439 | Pending |
| vpc_staged_c8 | RUNNING | 9 | 0.20954 | Pending |
| unet | RUNNING | 2 | 0.19484 | Pending |
| fno | COMPLETE | 9 | 0.23025 | 0.23544 |


## Reproduce the experiments

Use Python 3.11 or newer and a CUDA-enabled PyTorch installation for the optimized training runs. Point `SHARDS` at the external RealPDEBench Cylinder `hf_dataset/real` Arrow directory.

```bash
pip install -r requirements.txt
mkdir -p vendor
git clone https://github.com/AI4Science-WestlakeU/RealPDEBench.git vendor/RealPDEBench
git -C vendor/RealPDEBench checkout 06ba38d2db3e78edc2556ff7d9098119a42c2b07
git -C vendor/RealPDEBench apply "$PWD/configs/unet.patch"
```

The U-Net patch reproduces the run's import compatibility and CUDA Graph changes. FNO uses the pinned upstream architecture. The shared training protocol uses batch size 1, Adam at 0.0003, five-epoch patience, and validation checkpoint selection.

```bash
for channels in 8 16; do
  PYTHONPATH=code/joint:code/support python -m task3_cylinder_real.train \
    --model vpc --vpc-encoder spatial --vpc-latent-channels "$channels" \
    --vpc-latent-stride 4 --vpc-width 32 --vpc-layers 2 --vpc-heads 4 \
    --vpc-kl-weight 0.0001 --seed 525 --epochs 50 --test \
    --shards "$SHARDS" --splits data/splits --output "runs/joint_c$channels"
done

PYTHONPATH=code/staged:code/support python -m task3_cylinder_real.train \
  --model vpc --vpc-latent-channels 8 --seed 525 --epochs 50 --test \
  --init-vae initialization/staged_c8/best.pt \
  --vae-refine-epochs 1 --dynamics-pretrain-epochs 2 \
  --shards "$SHARDS" --splits data/splits --output runs/staged_c8

PYTHONPATH=code/unet:code/support:vendor/RealPDEBench python -m task3_cylinder_real.train \
  --model unet --realpdebench-config configs/unet.yaml --seed 525 --epochs 50 --test \
  --shards "$SHARDS" --splits data/splits --output runs/unet

PYTHONPATH=code/joint:code/support:vendor/RealPDEBench python -m task3_cylinder_real.train \
  --model fno --realpdebench-config configs/fno.yaml --seed 525 --epochs 10 --test \
  --shards "$SHARDS" --splits data/splits --output runs/fno
```

The staged initialization is the saved eight-channel VAE source checkpoint and its matching metadata. The completed FNO log comes from the earlier shared training run with the ten-epoch budget. The U-Net and joint and staged VPC entries are from the fifty-epoch runs.
