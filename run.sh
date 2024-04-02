#anomaly_ratio='0 0.05 0.1 0.15'
#query_strategy='entropy_sampling random_sampling margin_sampling least_confidence'
# 'capsule cable bottle carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper'
# 'capsule cable bottle carpet grid leather metal_nut pill screw tile toothbrush wood zipper'
gpu_id=$1

if [ $gpu_id == '0' ]; then
  class_name='pill screw tile toothbrush wood zipper metal_nut capsule cable bottle carpet grid leather'
  anomaly_ratio='0.0 0.1 0.2'
elif [ $gpu_id == '1' ]; then
  class_name='pill screw tile toothbrush wood zipper metal_nut capsule cable bottle carpet grid leather'
  anomaly_ratio='0.0 0.1 0.2'
elif [ $gpu_id == '99' ]; then
  class_name='pill'
  anomaly_ratio='0.0'
  gpu_id='1'
else
  echo "Invalid GPU ID. Please provide a valid GPU ID (0 or 1)."
fi


for c in $class_name
do
  for r in $anomaly_ratio
  do
    echo "sampling_method: $s class_name: $c anomaly_ratio: $r"      
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
    default_setting=./configs/default/mvtecad.yaml \
    model_setting=./configs/model/reconpatch.yaml \
    DATASET.class_name=$c \
    DATASET.params.anomaly_ratio=$r \
    DEFAULT.exp_name=test
    done
done
