
# MVTecAD 'capsule cable bottle carpet grid leather metal_nut pill screw tile toothbrush wood zipper'
# MVTecAD 'capsule hazelnut transistor cable bottle carpet grid leather metal_nut pill screw tile toothbrush wood zipper'
# VISA  'candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum'
# MVTecLoco 'breakfast_box  juice_bottle  pushpins  screw_bag  splicing_connectors'
# MPDD 'tubes metal_plate connector bracket_white bracket_brown bracket_black'
gpu_id=$1
#method_setting='coreinit patchcore reconpatch fastflow rd'
method_setting='coreinit'

if [ $gpu_id == '0' ]; then 
  class_name='candle capsules cashew chewinggum fryum'
  default_setting=./configs/default/visa.yaml
  anomaly_ratio='0.0'  
  Temperature='0.1'

elif [ $gpu_id == '1' ]; then 
  class_name='macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum'
  default_setting=./configs/default/visa.yaml
  anomaly_ratio='0.0'  
  Temperature='0.1'

elif [ $gpu_id == '2' ]; then #
  class_name='1 2 3'
  default_setting=./configs/default/btad.yaml
  anomaly_ratio='0.0'  
  Temperature='0.1'

elif [ $gpu_id == '3' ]; then 
  class_name='capsule hazelnut transistor cable bottle carpet grid leather metal_nut pill screw tile toothbrush wood zipper'
  default_setting=./configs/default/mvtecad.yaml
  anomaly_ratio='0.0'  
  Temperature='2'

elif [ $gpu_id == '99' ]; then
  class_name='03'
  default_setting=./configs/default/btad.yaml
  anomaly_ratio='0.0'
  method_setting='rd'
  Temperature='0.05'
  gpu_id='2'

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
    for r in $Temperature
    do
      echo "method_setting: $m class_name: $c Temperature: $r"      
      CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
      default_setting=$default_setting \
      model_setting=./configs/model/$m.yaml \
      DATASET.class_name=$c \
      DATAET.params.anomaly_ratio=$anomaly_ratio \
      DEFAULT.exp_name=BASELINE_focalloss_augment_norm_core_coreset_init_T_$r \
      MODEL.params.temperature=$r
      done
  done
done 