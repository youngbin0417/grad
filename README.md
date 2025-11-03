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

Training a CAML model is a two-step process. First, a baseline model must be trained using the *same technique* (ERM, EMA, or SWA) as the CAML model you intend to train. Then, the CAML model is trained using the outputs of this corresponding baseline model.

**Step 1: Train the Baseline Model**

Train the baseline model using the desired technique. The resulting model file (e.g., `baseline_erm.pth`, `baseline_ema.pth`, `baseline_swa.pth`) is required for the next step.

*   **Train a standard baseline (ERM):**
    ```bash
    python margin_loss.py --dataset waterbirds --train --type baseline
    ```

*   **Train a baseline with EMA:**
    ```bash
    python margin_loss_ema.py --dataset waterbirds --train --type baseline
    ```

*   **Train a baseline with SWA:**
    ```bash
    python margin_loss_swa.py --dataset waterbirds --train --type baseline --swa
    ```

**Step 2: Train the CAML Model**

Once the corresponding baseline model is trained, you can train the CAML model using the same technique.

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

Use `predict.py` to evaluate a trained model on the test set. You must specify the model type and the training technique.

**Example Commands:**

*   **Evaluate the standard CAML (ERM) model:**
    ```bash
    python predict.py --dataset waterbirds --type margin --technique erm
    ```

*   **Evaluate the baseline model trained with EMA:**
    ```bash
    python predict.py --dataset waterbirds --type baseline --technique ema
    ```

</br>
