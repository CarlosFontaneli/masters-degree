#!/usr/bin/env bash

PARAM_FILE="train_unet_medsam_params.json"

# 1) Run unet_train.py for each parameter set
echo "=== Running unet_train.py experiments ==="
UNET_LENGTH=$(jq '.unet_train | length' $PARAM_FILE)
for (( i=0; i<$UNET_LENGTH; i++ ))
do
  # Extract parameters using jq
  TRAIN_SIZE=$(jq -r ".unet_train[$i].train_size" $PARAM_FILE)
  EPOCHS=$(jq -r ".unet_train[$i].epochs" $PARAM_FILE)
  LR=$(jq -r ".unet_train[$i].lr" $PARAM_FILE)
  BATCH_SIZE=$(jq -r ".unet_train[$i].batch_size" $PARAM_FILE)
  OPTIMIZER=$(jq -r ".unet_train[$i].optimizer" $PARAM_FILE)
  LOSS_TYPE=$(jq -r ".unet_train[$i].loss_type" $PARAM_FILE)
  SCHEDULER=$(jq -r ".unet_train[$i].scheduler" $PARAM_FILE)
  AUGMENT=$(jq -r ".unet_train[$i].augment" $PARAM_FILE)
  IMAGE_SIZE=$(jq -r ".unet_train[$i].image_size" $PARAM_FILE)
  ACCUMULATE_GRAD_STEPS=$(jq -r ".unet_train[$i].accumulate_grad_steps" $PARAM_FILE)

  echo "Running UNet experiment #$((i+1)) with train_size=$TRAIN_SIZE, epochs=$EPOCHS, lr=$LR"
  
  CMD="python ./u-net/unet_train.py \
    --train_size $TRAIN_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --optimizer $OPTIMIZER \
    --loss_type $LOSS_TYPE \
    --scheduler $SCHEDULER \
    --augment $AUGMENT \
    --image_size $IMAGE_SIZE \
    --accumulate_grad_steps $ACCUMULATE_GRAD_STEPS"
  
  echo "Executing: $CMD"
  $CMD
done

# 2) Run medsam_fine_tune.py for each parameter set
echo "=== Running medsam_fine_tune.py experiments ==="
MEDSAM_LENGTH=$(jq '.medsam_fine_tune | length' $PARAM_FILE)
for (( j=0; j<$MEDSAM_LENGTH; j++ ))
do
  TRAIN_SIZE=$(jq -r ".medsam_fine_tune[$j].train_size" $PARAM_FILE)
  EPOCHS=$(jq -r ".medsam_fine_tune[$j].epochs" $PARAM_FILE)
  LR=$(jq -r ".medsam_fine_tune[$j].lr" $PARAM_FILE)
  BATCH_SIZE=$(jq -r ".medsam_fine_tune[$j].batch_size" $PARAM_FILE)
  OPTIMIZER=$(jq -r ".medsam_fine_tune[$j].optimizer" $PARAM_FILE)
  LOSS_TYPE=$(jq -r ".medsam_fine_tune[$j].loss_type" $PARAM_FILE)
  SCHEDULER=$(jq -r ".medsam_fine_tune[$j].scheduler" $PARAM_FILE)
  AUGMENT=$(jq -r ".medsam_fine_tune[$j].augment" $PARAM_FILE)
  ACCUMULATE_GRAD_STEPS=$(jq -r ".medsam_fine_tune[$j].accumulate_grad_steps" $PARAM_FILE)
  PRECISION=$(jq -r ".medsam_fine_tune[$j].precision" $PARAM_FILE)

  echo "Running MedSAM experiment #$((j+1)) with train_size=$TRAIN_SIZE, epochs=$EPOCHS, lr=$LR"
  
  CMD="python ./med-sam/medsam_fine_tune.py \
    --train_size $TRAIN_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --optimizer $OPTIMIZER \
    --loss_type $LOSS_TYPE \
    --scheduler $SCHEDULER \
    --augment $AUGMENT \
    --accumulate_grad_steps $ACCUMULATE_GRAD_STEPS \
    --precision $PRECISION"
  
  echo "Executing: $CMD"
  $CMD
done

echo "=== All experiments completed ==="
