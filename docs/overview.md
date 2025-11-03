# EfficientAD Standalone - 專案概覽

## 專案目標

從 anomalib 提取的**輕量化 EfficientAD 實作**，專為生產環境部署設計。這是一個簡化的獨立版本，專注於核心功能。

## 專案資料夾結構

```
MLExploration_efficientAD/
├── efficientad/                      # 核心模組
│   ├── models/
│   │   ├── torch_model.py           # 純 PyTorch 模型（生產用）
│   │   └── lightning_model.py       # PyTorch Lightning 模型（訓練用）
│   ├── data/
│   │   ├── dataclasses.py           # 資料類別定義
│   │   ├── transforms/              # 資料轉換
│   │   │   ├── center_crop.py       # 中心裁切
│   │   │   └── utils.py             # 轉換工具
│   │   └── utils/
│   │       └── download.py          # 資料下載工具
│   ├── preprocessing/
│   │   ├── pre_processor.py         # 前處理主程式
│   │   └── utils/
│   │       └── transform.py         # 前處理轉換
│   └── trainer.py                   # 訓練器主程式
│
├── src_run/                          # 執行腳本
│   ├── train_standalone.py          # 訓練主程式
│   ├── visualize_standalone.py      # 視覺化測試結果
│   ├── visualize_pdn_layers.py      # 視覺化 PDN 各層特徵
│   ├── inference_standalone.py      # 推論主程式
│   └── config_*.yaml                # 各模型的配置檔（anomalib CLI 用）
│
├── *.sh                              # Shell 執行腳本
│   ├── train_standalone.sh          # 訓練執行腳本
│   ├── visualize_standalone.sh      # 視覺化執行腳本
│   └── visualize_pdn_layers.sh      # PDN 層視覺化執行腳本
│
├── sh_backup/                        # 舊腳本備份
│   └── train_*.sh                   # anomalib CLI 訓練腳本
│
├── docs/                             # 專案文件
│   └── overview.md                  # 專案概覽
│
├── datasets/                         # 資料集（不納入 git）
│   └── VirtualSEM/
│       └── repeating/
│           ├── train/good/
│           ├── test/good/
│           ├── test/defect_type/
│           └── ground_truth/
│
├── pre_trained/                      # 預訓練模型（不納入 git）
│   ├── dinov2_vitb14_reg4_pretrain.pth
│   └── efficientad_pretrained_weights/
│
└── results_standalone/               # 訓練結果（不納入 git）
    └── EfficientAd/
        └── {dataset_name}/
            └── {category}_{model_size}/
                ├── model.pth
                ├── images/test/          # visualize_standalone.sh 輸出
                └── pdn_layers/test/      # visualize_pdn_layers.sh 輸出
```

## 程式碼關係圖

### 訓練流程
```
train_standalone.sh
    └─> train_standalone.py
            ├─> efficientad.trainer.EfficientADTrainer
            │       └─> efficientad.models.torch_model.EfficientAdModel
            └─> outputs: results_standalone/.../model.pth
```

### 視覺化測試結果流程
```
visualize_standalone.sh
    └─> visualize_standalone.py
            ├─> loads: results_standalone/.../model.pth
            ├─> efficientad.models.torch_model.EfficientAdModel
            └─> outputs: results_standalone/.../images/test/
```

### 視覺化 PDN 層流程
```
visualize_pdn_layers.sh
    └─> visualize_pdn_layers.py
            ├─> loads: results_standalone/.../model.pth
            ├─> PDNLayerExtractor (內部定義)
            ├─> efficientad.models.torch_model.EfficientAdModel
            └─> outputs: results_standalone/.../pdn_layers/test/
```

## 主要工作流程

### 1. 訓練 (`train_standalone.sh`)

```bash
./train_standalone.sh
```

**設定：**
- 訓練資料：`./datasets/VirtualSEM/repeating/train`
- 測試資料：`./datasets/VirtualSEM/repeating/test`
- 輸出路徑：`results_standalone/EfficientAd/VirtualSEM/repeating_medium/`
- 模型大小：Medium (M)
- 圖片尺寸：256x256
- 訓練週期：70
- Teacher 輸出通道：384

### 2. 視覺化測試結果 (`visualize_standalone.sh`)

```bash
./visualize_standalone.sh
```

**功能：**
- 載入訓練好的模型
- 使用 good 樣本計算 quantiles
- 產生異常熱圖（map_st + map_stae）
- 輸出視覺化：原圖 | GT Mask | map_st | map_stae | Final
- 測試資料：`./datasets/VirtualSEM_v3/repeating/test`
- 輸出路徑：`results_standalone/.../images/test/`

### 3. 視覺化 PDN 各層特徵 (`visualize_pdn_layers.sh`)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate anomalib
./visualize_pdn_layers.sh
```

**功能：**
- 顯示 8 層 Teacher-Student 特徵差異熱圖
- 顯示各層的感受野（Receptive Field）大小
- 包含顏色條和數值範圍
- 測試資料：`./datasets/VirtualSEM/repeating/test`
- 訓練資料：用於計算各層統計量（mean/std）
- 輸出路徑：`results_standalone/.../pdn_layers/test/`

## 技術細節

### 模型架構

**PDN (Patch Description Network)** - Teacher 和 Student 網路之間的知識蒸餾

8 層卷積結構：
```
conv1 (RF:4x4) → pool1 (RF:5x5) → conv2 (RF:11x11) → pool2 (RF:13x13) →
conv3 (RF:13x13) → conv4 (RF:21x21) → conv5 (RF:33x33) → conv6 (RF:33x33)
```

### 異常檢測方法

- **map_st**：Student-Teacher 特徵距離
- **map_stae**：Student-Teacher AutoEncoder 重建誤差
- **最終異常圖**：0.5 * map_st + 0.5 * map_stae

### 資料集格式

ImageFolder 結構：
```
VirtualSEM/
├── repeating/
│   ├── train/
│   │   └── good/
│   ├── test/
│   │   ├── good/
│   │   └── defect_type1/
│   └── ground_truth/
│       └── defect_type1/
│           └── *_mask.png
```

## 重要注意事項

### 資料集路徑不一致

- 訓練使用：`./datasets/VirtualSEM`
- 視覺化使用：`./datasets/VirtualSEM_v3` ⚠️

### 環境需求

- `visualize_pdn_layers.sh` 需要 conda 環境 `anomalib`
- 其他腳本直接使用系統 Python

## 模型大小選項

`EfficientAdModelSize` 提供三種模型大小：
- **SMALL (S)**：精簡模型
- **MEDIUM (M)**：預設平衡模型（目前配置）
- **DINO**：基於 DinoV2 的變體（圖片尺寸：896x896）
