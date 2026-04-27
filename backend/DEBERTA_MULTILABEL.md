# DeBERTa Multi-Label Emotion Pipeline

This is the high-accuracy path for SentiFlow's next model iteration.

## Train

Full fine-tuning:

```bash
cd backend
python train_deberta_multilabel.py --gpu --epochs 4 --batch 8 --grad-accum 2
```

Resource-friendly LoRA run:

```bash
cd backend
python train_deberta_multilabel.py --gpu --lora --epochs 4 --batch 8 --grad-accum 2
```

Smoke test:

```bash
cd backend
python train_deberta_multilabel.py --limit 200 --epochs 1 --batch 4
```

Outputs:

- `deberta_multilabel/model/`
- `deberta_multilabel/calibration.json`
- `deberta_multilabel/metrics.json`

The trainer maps GoEmotions rows into SentiFlow's 8 labels as multi-hot targets,
uses weighted BCE loss for class imbalance, fits temperature scaling on the
validation split, then chooses one threshold per emotion by validation F1.

## Compare Accuracy And Speed

```bash
cd backend
python compare_emotion_models.py \
  --deberta-model ./deberta_multilabel/model \
  --deberta-calibration ./deberta_multilabel/calibration.json \
  --limit 1000
```

This evaluates the local DeBERTa model and `SamLowe/roberta-base-go_emotions-onnx`
with `onnx/model_quantized.onnx`, then writes `model_comparison.json`.

## Use DeBERTa In The API

After training:

```bash
cd backend
SENTIFLOW_ANALYZER=deberta_multilabel \
SENTIFLOW_DEBERTA_MODEL=./deberta_multilabel/model \
SENTIFLOW_DEBERTA_CALIBRATION=./deberta_multilabel/calibration.json \
python main.py
```

The API response includes calibrated emotion scores, `active_emotions`, and an
`uncertain` flag when no emotion clears its calibrated threshold.
