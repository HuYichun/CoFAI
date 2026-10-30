# ProTRAC

M2474：在 ADE20K-C 语义分割和 AG-ReID-C 跨视角重识别上进行在线特征校准。

## 环境

请先按照仓库根目录 [README.md](../../README.md) 完成 Poetry 环境安装。
所有命令均从 CoFAI 根目录执行，并使用 `poetry run python`。

## 数据与权重

资源统一放在 [CoFAI 共享站](https://medialab.sjtu.edu.cn/files/CoFAI-share/)。

| 资源 | 下载 |
|---|---|
| ADE20K | [ADE20K.zip](https://medialab.sjtu.edu.cn/files/CoFAI-share/data/ADE20K.zip) |
| DINOv3 ViT-L/16 | [主干网络权重](https://medialab.sjtu.edu.cn/files/CoFAI-share/weights/dinov3/backbone/dinov3_vitl16_pretrain_lvd1689m.safetensors) |
| ADE20K 线性分割头 | [分割头权重](https://medialab.sjtu.edu.cn/files/CoFAI-share/weights/dinov3/semseg_head/dinov3_vitl16_semseg_ade20k_linear_head.pth) |
| AG-ReID | [AG-ReID.zip](https://medialab.sjtu.edu.cn/files/CoFAI-share/data/AG-ReID.zip) |
| VDT | [模型权重](https://medialab.sjtu.edu.cn/files/CoFAI-share/weights/protrac/agreid/vdt.pth) |
| 分割源统计 | [全局](https://medialab.sjtu.edu.cn/files/CoFAI-share/weights/protrac/ade20k/global_stats.pt)、[类别](https://medialab.sjtu.edu.cn/files/CoFAI-share/weights/protrac/ade20k/class_stats.pt)、[区域](https://medialab.sjtu.edu.cn/files/CoFAI-share/weights/protrac/ade20k/region_stats.pt) |
| 重识别源统计 | [源统计](https://medialab.sjtu.edu.cn/files/CoFAI-share/weights/protrac/agreid/stats.pt) |

准备后的目录：

```text
data/
  ADEChallengeData2016/
    images/validation/
    annotations/validation/
  AG-ReID/
    query_all_c0/
    query_all_c3/
    bounding_box_test_all_c0/
    bounding_box_test_all_c3/
weights/
  dinov3/
    backbone/dinov3_vitl16_pretrain_lvd1689m.safetensors
    semseg_head/dinov3_vitl16_semseg_ade20k_linear_head.pth
  protrac/
    ade20k/{global_stats,class_stats,region_stats}.pt
    agreid/{vdt.pth,stats.pt}
```

源统计使用提供的文件，配置中的路径可自行修改。

## 生成扰动数据

```bash
poetry run python examples/protrac/generate_ade20k_c.py
poetry run python examples/protrac/generate_agreid_c.py
```

默认扰动强度为 5：ADE20K 的 15 种扰动保存到 `data/ADE20K_val-c/`；AG-ReID 的 6 种扰动保存到 `data/AG-ReID/query_corruption/`。ADE20K 复用原标签，AG-ReID 仅扰动查询图像，图库使用原图。生成器固定随机种子为 `20260313`，跳过已有文件；可用 `--corruptions` 选择域。

## 运行

```bash
poetry run python examples/protrac/run_eval_semseg.py \
  --config examples/protrac/config/ade20k_c.yaml

poetry run python examples/protrac/run_eval_reid.py \
  --config examples/protrac/config/agreid.yaml
```

单域运行：

```bash
poetry run python examples/protrac/run_eval_semseg.py --corruptions gaussian_noise
poetry run python examples/protrac/run_eval_reid.py --corruptions motion_blur
```

分割逐图更新，重识别每批处理 64 张图像并更新一次统计；每个域、每个协议开始时重置在线统计。模型使用 CoFAI 的 `CommonFeatureCodecModel` 和 `BypassLatentCodec`。结果及实际环境版本保存到 `logs/protrac/{semseg,reid}/results.json`。

## ProTRAC 结果

| 任务 | 域数 | mIoU (%) | mAP (%) | Rank-1 (%) |
|---|---:|---:|---:|---:|
| ADE20K-C | 15 | 44.20 | — | — |
| AG-ReID-C，AG | 6 | — | 41.01 | 46.41 |
| AG-ReID-C，GA | 6 | — | 27.44 | 30.04 |

以上为 M2474 提案的域平均结果，扰动强度均为 5。分割在切分点 Slot 23 校准，重识别在切分点 Slot 11 校准。
