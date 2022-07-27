# Curricular Transfer Learning
Dependencies and configuration:
```sh
pip install -r requirements.txt
accelerate config
```
EXAMPLE:
```
compute_environment: LOCAL_MACHINE
deepspeed_config:
 gradient_accumulation_steps: 1
 gradient_clipping: 1.0
 offload_optimizer_device: none
 offload_param_device: none
 zero3_init_flag: false
 zero_stage: 2
distributed_type: DEEPSPEED
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
```

Now run:
```sh
ln -s -r ../multiwoz/data/MultiWOZ_2.2/ data/multiwoz
ln -s -r ../multiwoz/db data/db
python process/multiwoz.py
python run.py
```
