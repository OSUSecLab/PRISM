# Function Name Recovery via Self-Transformative and Variable-Enrichment Approaches

## Repository Contents

## Running Steps
### Decompile Binaries
The related scripts are in the folder [`dataset_generation`](dataset_generation).

1. Start by specifying the `output_dir` in [`decomp_for_unstripped.py`](dataset_generation/decomp_for_unstripped.py) and [`decomp_for_stripped.py`](dataset_generation/decomp_for_stripped.py) to define where the results will be saved.
2. Use [`parallel_decomp.py`](scripts/decompilation/parallel_decomp.py) to decompile your binaries.
    - Use the `-u` option for unstripped binaries
    - Use the `-s` option for stripped binaries.


### Process Decompiled Function Code
The related scripts are in the folder [`deduplication`](deduplication).

For generating a dataset (`-d`), we need combine the stripped functions with unstripped functions using DWARF.
For prediction of stripped binary (`-p`), it only needs to output the test set file containing all masked functions.

#### Combine the stripped funcions with unstripped funcions
For generating a dataset:
```bash
python deduplication/process_decompiled_data.py -d \
    --unstripped_path YOUR_UNSTRIPPED_DECOMP_DATA_PATH \
    --stripped_path YOUR_STRIPPED_DECOMP_DATA_PATH \
    --output_dir YOUR_OUTPUT_DIR
```

For prediction of stripped binary:
```bash
python deduplication/process_decompiled_data.py -p \
    --stripped_path YOUR_STRIPPED_DECOMP_DATA_PATH \
    --output_dir YOUR_OUTPUT_DIR
```

#### Divide Dataset
Use [`divide_dataset.py`](deduplication/divide_dataset.py) to split the processed data into training, test, and validation sets. Duplicate entries are removed from the dataset based on the function name and function body. For more details, please refer to our paper.
```bash
python deduplication/divide_dataset.py \
    --input_dir YOUR_PROCESSED_DECOMP_DIR \
    --output_dir YOUR_OUTPUT_DIR \
    --arch YOUR_BINARY_ARCH \
    --opt YOUR_BINARY_OPT
```

#### Self-transformation
Use [`rand_replace.py`](replace/rand_replace.py) to conduct self-transformation process. The default directory of dataset is dataset/ARCH/OPT.

```bash
python replace/rand_replace.py \
    --arch YOUR_BINARY_ARCH \
    --opt YOUR_BINARY_OPT
```

### Fine-tune and Predict
This step is implemented based on [alpaca-lora](https://github.com/tloen/alpaca-lora/tree/main).

We use the Code Llama prompt format (see [templates/codellama.json](scripts/templates/codellama.json)).
The related scripts are in the folder [`scripts/`](scripts/).

#### Fine-tune vLLM
To fine-tune the vLLM model using your decompiled training set, use the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python script/finetune_codellama_var_dora.py \
    --arch YOUR_BINARY_ARCH \
    --opt YOUR_BINARY_OPT
```

#### Fine-tune fLLM
To fine-tune the vLLM model using your decompiled training set, use the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python script/finetune_codellama_dora.py \
    --arch YOUR_BINARY_ARCH \
    --opt YOUR_BINARY_OPT
```



#### vLLM Predict
To generate predictions using a fine-tuned vLLM model, execute the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python script/codellama_script_var_single.py \
    --arch YOUR_BINARY_ARCH \
    --opt YOUR_BINARY_OPT
```

#### Variable Replacement
To replace the output of vLLM, execute the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python script/var_replace/replace.py \
    --arch YOUR_BINARY_ARCH \
    --opt YOUR_BINARY_OPT
```

#### fLLM Predict
To generate predictions using a fine-tuned fLLM model, execute the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python script/codellama_script_dora_single.py \
    --arch YOUR_BINARY_ARCH \
    --opt YOUR_BINARY_OPT
```

### Evaluate
The related scripts are in the folder [`script/evaluation`](scripts/evaluation).

1. Extract the function name from the results and divide them into tokens:
    ```bash
    python evaluation/divide_function_name.py \
        --input_file YOUR_PRED_RESULTS_FILE \
        --output_dir YOUR_OUTPUT_DIR
    ```


2. To calculate the precision, recall and F1 score, use the following command:
    ```bash
    python evaluation/calculate_score.py \
        --input_file YOUR_PROCESSED_RESULTS_FILE
    ```


## License
This project is licensed under the Apache License 2.0 (See [LICENSE](LICENSE)).

This project includes other components distributed under the Apache License 2.0:

- alpaca-lora (See [scripts/model_training/LICENSE](scripts/model_training/LICENSE))
