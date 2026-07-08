#!/usr/bin/env bash
#
# Train a Talk2DINO projection for DINOv2 ViT-S/14-reg (CLIP ViT-B/16 text -> 384-d DINO space).
#
# Pipeline (see README "Feature Extraction" + "Training"):
#   1. extract DINOv2 ViT-S image features for COCO-2014 train + val
#   2. extract CLIP ViT-B/16 text features for the captions (backbone-independent --
#      set SKIP_TEXT=1 to reuse features from a previous ViT-B/L run)
#   3. train the projection MLP (configs/vits_mlp_infonce.yaml)
#
# Run from the Talk2DINO repo root, in the hovsg conda env:
#   conda run -n hovsg bash train_vits.sh
#
# Override any path via environment variables, e.g.
#   COCO_DIR=/data/coco OUT_DIR=/data/coco2014_s14 bash train_vits.sh
#
set -euo pipefail

# --- configuration -----------------------------------------------------------
MODEL="${MODEL:-dinov2_vits14_reg}"          # reg variant, to match the rest of the workspace
CONFIG="${CONFIG:-configs/vits_mlp_infonce.yaml}"
COCO_DIR="${COCO_DIR:-../coco}"              # holds train2014/ and val2014/ image folders
CAPTION_DIR="${CAPTION_DIR:-$COCO_DIR/annotations}"   # COCO caption JSONs
OUT_DIR="${OUT_DIR:-../coco2014_s14}"        # where the extracted feature .pth files go
RESIZE_DIM="${RESIZE_DIM:-448}"
CROP_DIM="${CROP_DIM:-448}"
IMG_BATCH="${IMG_BATCH:-64}"
TXT_BATCH="${TXT_BATCH:-256}"
SKIP_TEXT="${SKIP_TEXT:-0}"                  # set to 1 to reuse existing text features
NAME_PEDIX="${NAME_PEDIX:-}"                 # suffix for the output weights name, e.g. 'noreg'

TRAIN_ANN="$CAPTION_DIR/captions_train2014.json"
VAL_ANN="$CAPTION_DIR/captions_val2014.json"
TRAIN_PTH="$OUT_DIR/train.pth"
VAL_PTH="$OUT_DIR/val.pth"

echo "== Talk2DINO ViT-S training =="
echo "  model      : $MODEL"
echo "  config     : $CONFIG"
echo "  coco dir   : $COCO_DIR"
echo "  captions   : $CAPTION_DIR"
echo "  out dir    : $OUT_DIR"
echo "  resolution : ${RESIZE_DIM}/${CROP_DIM}  (resize/crop)"
echo

# --- sanity checks -----------------------------------------------------------
for f in "$TRAIN_ANN" "$VAL_ANN"; do
    [[ -f "$f" ]] || { echo "ERROR: caption file not found: $f" >&2; exit 1; }
done
[[ -d "$COCO_DIR/train2014" ]] || { echo "ERROR: $COCO_DIR/train2014 not found" >&2; exit 1; }
[[ -d "$COCO_DIR/val2014"   ]] || { echo "ERROR: $COCO_DIR/val2014 not found"   >&2; exit 1; }
mkdir -p "$OUT_DIR"

# --- 1. image features -------------------------------------------------------
echo "== [1/3] Extracting DINOv2 image features =="
python dino_extraction_v2.py --ann_path "$VAL_ANN"   --out_path "$VAL_PTH"   --data_dir "$COCO_DIR" \
    --model "$MODEL" --resize_dim "$RESIZE_DIM" --crop_dim "$CROP_DIM" --batch_size "$IMG_BATCH" \
    --extract_avg_self_attn --extract_disentangled_self_attn
python dino_extraction_v2.py --ann_path "$TRAIN_ANN" --out_path "$TRAIN_PTH" --data_dir "$COCO_DIR" \
    --model "$MODEL" --resize_dim "$RESIZE_DIM" --crop_dim "$CROP_DIM" --batch_size "$IMG_BATCH" \
    --extract_avg_self_attn --extract_disentangled_self_attn

# --- 2. text features (CLIP ViT-B/16, backbone-independent) -------------------
if [[ "$SKIP_TEXT" == "1" ]]; then
    echo "== [2/3] Skipping text feature extraction (SKIP_TEXT=1) =="
else
    echo "== [2/3] Extracting CLIP text features =="
    python text_features_extraction.py --ann_path "$TRAIN_PTH" --batch_size "$TXT_BATCH"
    python text_features_extraction.py --ann_path "$VAL_PTH"   --batch_size "$TXT_BATCH"
fi

# --- 3. train ----------------------------------------------------------------
echo "== [3/3] Training the projection =="
TRAIN_ARGS=(--model "$CONFIG" --train_dataset "$TRAIN_PTH" --val_dataset "$VAL_PTH")
WEIGHTS_STEM="$(basename "$CONFIG" .yaml)"
if [[ -n "$NAME_PEDIX" ]]; then
    TRAIN_ARGS+=(--name_pedix "$NAME_PEDIX")   # train.py appends _<pedix> to the weights name
    WEIGHTS_STEM="${WEIGHTS_STEM}_${NAME_PEDIX}"
fi
python train.py "${TRAIN_ARGS[@]}"

WEIGHTS_OUT="weights/${WEIGHTS_STEM}.pth"
echo
echo "Done. Trained projection saved to: $WEIGHTS_OUT"
echo "To use it in rl_model_wrappers, vendor it alongside the others, e.g.:"
echo "  cp $WEIGHTS_OUT ../rl_model_wrappers/src/rl_model_wrappers/weights/talk2dino/"
