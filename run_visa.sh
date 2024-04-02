#anomaly_ratio='0 0.05 0.1 0.15'
#query_strategy='entropy_sampling random_sampling margin_sampling least_confidence'
# 'capsule cable bottle carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper'
# 'capsule cable bottle carpet grid leather metal_nut pill screw tile toothbrush wood zipper'
# candle  capsules  cashew  chewinggum  fryum  macaroni1  macaroni2  pcb1  pcb2  pcb3  pcb4  pipe_fryum  split_csv
gpu_id=$1

if [ $gpu_id == '0' ]; then
  class_name='candle  capsules  cashew  chewinggum  fryum  macaroni1'
  anomaly_ratio='0.0'
elif [ $gpu_id == '1' ]; then
  class_name='macaroni2  pcb1  pcb2  pcb3  pcb4  pipe_fryum'
  anomaly_ratio='0.0'
elif [ $gpu_id == '99' ]; then
  class_name='juice_bottle'
  anomaly_ratio='0.0'
  gpu_id='0'
else
  echo "Invalid GPU ID. Please provide a valid GPU ID (0 or 1)."
fi


for c in $class_name
do
  for r in $anomaly_ratio
  do
    echo "sampling_method: $s class_name: $c anomaly_ratio: $r"      
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
    default_setting=./configs/default/mvtecloco.yaml \
    model_setting=./configs/model/proxy.yaml \
    DATASET.class_name=$c \
    DATASET.params.anomaly_ratio=$r \
    DEFAULT.exp_name=test_loco
    done
done
