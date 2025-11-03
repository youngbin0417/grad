# Mitigating Biases in Blackbox Feature Extractors for Image Classification Tasks

본 리포지토리는 졸업 프로젝트를 위한 수정 된 파일들의 리포지토리입니다
	
</br>

## Dataset

* Download the CelebA, and Waterbirds datasets. Extract their features from a pretrained feature extractor and store them. Note the folder path and update it in `utils/config.py`. Recall that the validation set has to be group-balanced. For CelebA, store the folder in the `celeba_path` attribute of `utils/config.py` (similarly for Waterbirds). The embeddings should be stored in `{split}_feats.npy`, whereas the bias and target labels should be stored in `{split}_bias.npy` and `{split}_target.npy` respectively, inside the folder. 

</br>

## 

## Usage

All model paths are now configured in `utils/config.py` to avoid overwriting files.

### Training

Training a CAML model is a two-step process. First, a standard ERM baseline model must be trained. This baseline model is then used by all CAML variants (ERM, EMA, SWA) to calculate margins.

**Step 1: Train the Standard ERM Baseline Model**

This model is trained using standard Empirical Risk Minimization. The resulting model file (`baseline_erm.pth`) is required for the next step, regardless of whether you intend to train a CAML model with ERM, EMA, or SWA.

```bash
# Train a standard baseline model
python margin_loss.py --dataset waterbirds --train --type baseline
```

**Step 2: Train the CAML Model (with ERM, EMA, or SWA)**

Once the standard ERM baseline model is trained, you can train the CAML model using your desired technique. The CAML model will internally use the `baseline_erm.pth` for margin calculation.

*   **Train a standard CAML model (ERM):**
    ```bash
    python margin_loss.py --dataset waterbirds --train --type margin
    ```

*   **Train a CAML model with EMA:**
    ```bash
    python margin_loss_ema.py --dataset waterbirds --train --type margin
    ```

*   **Train a CAML model with SWA:**
    ```bash
    python margin_loss_swa.py --dataset waterbirds --train --type margin --swa
    ```

### Prediction

Use `predict.py` to evaluate and compare the performance of the `baseline` and `CAML` models for a specific training technique. The script will automatically find the corresponding model files and display their results in a comparison table.

**Example Commands:**

*   **Compare models trained with standard ERM:**
    ```bash
    python predict.py --dataset waterbirds --technique erm
    ```

*   **Compare models trained with EMA:**
    ```bash
    python predict.py --dataset waterbirds --technique ema
    ```

*   **Compare models trained with SWA:**
    ```bash
    python predict.py --dataset waterbirds --technique swa
    ```

</br>
