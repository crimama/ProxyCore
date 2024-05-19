
# MVTecAD 'capsule cable bottle carpet grid leather metal_nut pill screw tile toothbrush wood zipper'
# MVTecAD 'capsule hazelnut transistor cable bottle carpet grid leather metal_nut pill screw tile toothbrush wood zipper'
# VISA  'candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum'
# MVTecLoco 'breakfast_box  juice_bottle  pushpins  screw_bag  splicing_connectors'
# MPDD 'tubes metal_plate connector bracket_white bracket_brown bracket_black'
gpu_id=$1
method_setting='reconpatch'

if [ $gpu_id == '0' ]; then #visa 
  class_name='capsule hazelnut transistor cable bottle carpet grid leather metal_nut pill screw tile toothbrush wood zipper'
  default_setting=./configs/default/mvtecad.yaml
  anomaly_ratio='0.0'  

elif [ $gpu_id == '1' ]; then #visa 
  class_name='candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum'
  default_setting=./configs/default/visa.yaml
  anomaly_ratio='0.0'  

elif [ $gpu_id == '2' ]; then #MVTECAD 
  class_name='metal_plate'
  default_setting=./configs/default/mpdd.yaml
  anomaly_ratio='0.0'  

elif [ $gpu_id == '3' ]; then #MVTECAD 
  class_name='tile toothbrush wood zipper'
  default_setting=./configs/default/mvtecad.yaml
  anomaly_ratio='0.0'  
  Temperature='0.05'

elif [ $gpu_id == '99' ]; then
  class_name='1'
  default_setting=./configs/default/btad.yaml
  anomaly_ratio='0.0'  
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
    for r in $anomaly_ratio
    do
      echo "method_setting: $m class_name: $c anomaly_ratio: $r"      
      CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
      default_setting=$default_setting \
      model_setting=./configs/model/$m.yaml \
      DATASET.class_name=$c \
      DEFAULT.exp_name=BASELINE
      done
  done
done 