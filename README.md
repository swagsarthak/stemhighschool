# stemhighschool

Fine-tune a BLIP-based model on the ScienceQA dataset to build a high-school STEM Q&A bot that can score multiple-choice answers (with optional images).

## Setup
- Install dependencies (Python 3.10+ recommended): `pip install torch torchvision transformers datasets pillow scikit-learn wandb tqdm`
- (Optional) Log in to Hugging Face if the BLIP weights require auth: `huggingface-cli login`
- GPU is recommended; the code will fall back to CPU if CUDA is unavailable.

## Training
- Configure hyperparameters in `config.py` (adjust dataset splits for full training runs).
- Start fine-tuning (offline W&B logging by default):
  ```bash
  python main.py train
  ```
- The best checkpoint is saved to `outputs/saved_models/scienceqa-blip-mc-best.pt`.
  If the directory does not exist it will be created automatically.

## Evaluation
- Evaluate the saved checkpoint on the test split:
  ```bash
  python main.py evaluate
  ```

## Inference
- Ask a custom question with your own answer options; wrap multi-word options in quotes. An image is optional (blank image is used otherwise):
  ```bash
  python main.py predict \
    --question "What is the chemical symbol for water?" \
    --choices "H2O" "CO2" "NaCl" "O2"
  # with an image
  python main.py predict \
    --question "What does this graph show?" \
    --choices "Linear growth" "Exponential growth" "Logarithmic growth" \
    --image_path path/to/plot.png
  ```

## Notes
- `NUM_CHOICES` is capped at 5 by default to match ScienceQA. Update `config.py` if you need more options.
- W&B runs in offline mode; sync later with `wandb sync` if desired.
