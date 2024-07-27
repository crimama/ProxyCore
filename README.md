# ProxyCore: Advancing Industrial Anomaly Localization with Coreset based Proxy Metric Learning
- Anomaly Detection Framework for Noisy unsupervised learning 
- Now ongoing 


# Environments

NVIDIA pytorch docker [ [link](https://github.com/ufoym/deepo) ]

```bash
docker pull ufoym/deepo:pytorch
```

## Requirements
[requirements.sh](requirements.sh)

```bash
bash requirements.sh
```
# Run Patchcore
```bash
gpu_id=$1

if [ $gpu_id == '0' ]; then
  class_name='capsule'
  anomaly_ratio='0.0'
  sampling_method='identity'
elif [ $gpu_id == '1' ]; then
  class_name='capsule'
  anomaly_ratio='0.1'
  sampling_method='identity'  
else
  echo "Invalid GPU ID. Please provide a valid GPU ID (0 or 1)."
fi

for s in $sampling_method
  do
  for c in $class_name
  do
    for r in $anomaly_ratio
    do
      echo "sampling_method: $s class_name: $c anomaly_ratio: $r"      
      CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
      default_setting=./configs/default/mvtecad.yaml \
      model_setting=./configs/model/rd.yaml \
      DATASET.class_name=$c \
      DATASET.params.anomaly_ratio=$r \
      MODEL.params.sampling_method=$s
      done
  done
done


```