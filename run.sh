
# MVTecAD 'capsule cable bottle carpet grid leather metal_nut pill screw tile toothbrush wood zipper'
# VISA  'candle  capsules  cashew  chewinggum  fryum  macaroni1  macaroni2  pcb1  pcb2  pcb3  pcb4  pipe_fryum'
# MVTecLoco 'breakfast_box  juice_bottle  pushpins  screw_bag  splicing_connectors'
gpu_id=$1
method_setting='patchcore proxy reconpatch fastflow rd'

if [ $gpu_id == '0' ]; then #visa 
  class_name='breakfast_box juice_bottle  pushpins  screw_bag  splicing_connectors'
  default_setting=./configs/default/mvtecloco.yaml
  method_setting='proxy reconpatch'
  anomaly_ratio='0.2'

elif [ $gpu_id == '1' ]; then #visa 
  class_name='breakfast_box juice_bottle  pushpins  screw_bag  splicing_connectors'
  default_setting=./configs/default/mvtecloco.yaml
  method_setting='reconpatch'
  anomaly_ratio='0.0 0.1'

elif [ $gpu_id == '2' ]; then #visa 
  class_name='splicing_connectors'
  default_setting=./configs/default/mvtecloco.yaml
  method_setting='reconpatch'
  anomaly_ratio='0.0'

elif [ $gpu_id == '3' ]; then #visa 
  class_name='splicing_connectors'
  default_setting=./configs/default/mvtecloco.yaml
  method_setting='reconpatch'
  anomaly_ratio='0.1'

elif [ $gpu_id == '99' ]; then
  class_name='screw'
  default_setting=./configs/default/mvtecad.yaml
  anomaly_ratio='0.0'
  method_setting='proxy'
  gpu_id='0'

elif [ $gpu_id == '77' ]; then
  class_name='toothbrush'
  default_setting=./configs/default/mvtecad.yaml
  anomaly_ratio='0.0 0.1 0.2'
  method_setting='proxy'
  gpu_id='1'

else
  echo "Invalid GPU ID. Please provide a valid GPU ID (0 or 1)."
fi

for m in $method_setting
do
  for c in $class_name
  do
    for r in $anomaly_ratio
    do
      echo "method_setting: $m class_name: $c anomaly_ratio: $r"      
      CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
      default_setting=$default_setting \
      model_setting=./configs/model/$m.yaml \
      DATASET.class_name=$c \
      DATASET.params.anomaly_ratio=$r \
      DEFAULT.exp_name=BaseDataset

      done
  done
done 