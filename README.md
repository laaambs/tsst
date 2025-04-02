# Tree-Searched Self-Training (TSST)

This repository implements a **Tree-Searched Self-Training (TSST)** framework for enhancing multi-modal reasoning, with a focus on the **A-OKVQA** dataset. The pipeline is divided into two major stages: **Warm-up** and **Self-Training**.

---

## 📥 Dataset Download

Before proceeding with the training, ensure you have downloaded the necessary datasets. You can obtain the datasets from the following links:

- **Visual Commonsense Reasoning (VCR)**: Download the dataset from [Visual Commonsense](https://visualcommonsense.com/).
- **Science Question Answering (ScienceQA)**: Download the dataset from [ScienceQA](https://scienceqa.github.io/).

These datasets are essential for training and evaluating the models in this repository.

---

## 🛠️ Environment Setup

This project relies on the LLaVA environment for training. The LLaVA codebase is included in this repository under `tsst/models/LLaVA`. Our value model and other modifications are built on top of this codebase, and the training scripts will utilize this code. For environment setup, please follow the setup instructions from the [LLaVA repository](https://github.com/haotian-liu/LLaVA).

## 🚀 Training Configuration

We use the training scripts and methodology from the LLaVA project for our training process. The training scripts are located in the LLaVA codebase within our repository.

### Training Scripts
- For SFT training of the policy model: `tsst/models/LLaVA/scripts/v1_5/finetune_policy_lora.sh`
- For SFT training of the value model: `tsst/models/LLaVA/scripts/v1_5/finetune_value_lora.sh`

### Configuration Parameters
Each training script requires the following configuration parameters at the beginning:

```bash
# Required configuration
PROJECT_ROOT=           # Absolute path to this project
VISION_MODEL_PATH=      # Path to clip-vit-large-patch14-336 used in LLaVA
BASE_MODEL_PATH=        # Path to the base model to be fine-tuned (policy or value model)
OUTPUT_DIR=             # Path where the trained model will be saved
DATA_PATH=              # Path to the JSON dataset for LLaVA training
IMAGE_DIR=              # Path to the directory containing images for LLaVA training
```

```bash
# Example usage
bash models/LLaVA/scripts/v1_5/finetune_policy_lora.sh
bash models/LLaVA/scripts/v1_5/finetune_value_lora.sh
```   


## 🧊 Stage 1: Warm-up

1. **Chain-of-Thought Data Generation**  
   Use **LLaMA 70B** to generate Chain-of-Thought (CoT) rationales for [A-OKVQA](https://github.com/allenai/aokvqa) questions. 

   The dataset is available [here](https://drive.google.com/drive/folders/1tns4mRmv2Lf8Fz8pNN6WLy6hsvnKvbf3?usp=drive_link).

   Save the generated CoT data in `dataset/llava_aokvqa_sft_policy.json`.



2. **Value Estimation and Scoring**  
   Apply a simple linear scoring mechanism to assign value scores to the generated CoT samples. Each step in the reasoning chain receives an incremental score, with the final answer receiving the highest score. The scoring results can be found in the [google drive link](https://drive.google.com/drive/folders/1tns4mRmv2Lf8Fz8pNN6WLy6hsvnKvbf3?usp=drive_link).

   Save the scored CoT data in `dataset/llava_aokvqa_sft_value.json`.
   Example:
   ```
   Q: What sport is being played in this image?
   
   Step 1: I can see players wearing protective gear including helmets and pads (Score: 0.25)
   Step 2: The players are holding long sticks with nets at the end (Score: 0.50) 
   Step 3: The field has marked lines and goals at each end (Score: 0.75)
   Final Answer: The sport being played is lacrosse (Score: 1.0)
   ```

3. **Warm-up Model Training**  
   Fine-tune two models in parallel:
   - **Policy Model 0**: LLaVA fine-tuned on CoT data.
   - **Value Model 0**: LLaVA extended with an additional linear layer for value prediction, trained on scored CoT data.
   
   Here we use the `finetune_policy_lora.sh` and `finetune_value_lora.sh` scripts to train the policy and value models.

   First, fill in the configuration parameters in the training scripts.

   Then, run the training scripts.
   ```bash
   # Train policy model
   bash models/LLaVA/scripts/v1_5/finetune_policy_lora.sh

   # Train value model
   bash models/LLaVA/scripts/v1_5/finetune_value_lora.sh
   ```

---

## 🔁 Stage 2: Tree-Searched Self-Training

1. **Sampling from Policy**  
   Use **Policy Model 0**, guided by **Value Model 0**, to sample new CoT trajectories and answers. 

   Configuration for this process can be found in the `config/eval/{dataset}_beam.yaml` file. This configuration file allows you to customize various parameters for the sampling/evaluation process. Below is an explanation of the key parameters you can configure:

   - `num_beams`: Sets the number of beams for beam search, which affects the diversity of the generated samples.
   - `max_length`: Specifies the maximum length of the generated sequences.
   - `do_sample`: A boolean indicating whether to use sampling; if false, greedy decoding is used.
   - `temperature`: Adjusts the randomness of the sampling process; higher values lead to more diverse outputs.
   - `num_return_sequences`: Determines how many sequences to return for each input.
   - `task`: Defines the specific task type for the evaluation, such as "multiple_choice_steps_beam".
   - `dataset_name`: The name of the dataset file used for evaluation.
   - `dataset_root`: The root directory where the dataset is stored.
   - `data_subset`: Specifies a subset of the data to be used.
   - `data_partition`: Indicates the partition of the data to be used, such as a specific range of data.
   - `sample_size`: The number of samples to be used in the evaluation.
   - `seed`: Sets the random seed for reproducibility.
   - `branch`: Controls the branching factor in the sentence-level beam search, influencing the exploration of different paths.
   - `n_consistency`: Specifies the number of consistent samples required for a valid output.
   - `batch_size`: Determines the number of samples processed in each batch during evaluation.
   - `models`: A list of models to be used, each with specific configurations:
     - `policy_model`: Path to the policy model used for sampling.
     - `value_model`: Path to the value model that guides the sampling process.
     - `tag`: A tag for identifying the model configuration.
     - `run_flag`: A boolean indicating whether to run this model configuration.
     - `use_value`: A boolean indicating whether to use the value model for guidance.

   Ensure that you have the correct paths and settings configured in your YAML file to successfully run the evaluation. For verification purposes, we will use LLaMA-3 70B, which needs to be configured in config/llm_verifier_config.yaml.

   ```bash
   # Edit the CONFIG variable in scripts/run_eval_scienceqa.sh to point to your configured YAML file
   # For example: CONFIG=config/eval/vcr_beam.yaml
   
   # Then run the evaluation script
   bash scripts/run_eval_scienceqa.sh
   ```

2. **Verified Results**  
   The sampling script from step 1 will directly generate verified results in the `output/verified_data` directory. These results have already been evaluated and validated through the sampling process in 1.

3. **Dataset Construction**  
   Construct two datasets:
   - **D+**: Correctly labeled responses for policy fine-tuning
   - **Dv**: Corresponding value-labeled data for value model fine-tuning

   ```bash
   python utils/assign_value_tree.py \
       --input output/verified_data/results.jsonl \
       --output output/value_data/value_labeled.jsonl
   ```
   Then convert jsonl to LLaVA training format.

4. **Supervised Fine-tuning**  
   Fine-tune models on the newly constructed datasets:

   ```bash
   # Modify the parameters according to the warm-up parameter settings before fine-tuning
   bash models/LLaVA/scripts/v1_5/finetune_policy_lora.sh
   bash models/LLaVA/scripts/v1_5/finetune_value_lora.sh
   ```  

5. **Iterative Training**  
   Repeat steps 1 to 4 to continually refine the policy and value models.

---

## 🚀 Stage 3: Evaluation and Test-Time Scaling

In this stage, you can perform direct evaluation and test-time scaling using the provided scripts. Follow the instructions below to run these evaluations.

1. **Direct Evaluation**  
   This step involves evaluating the models directly on the test dataset. Ensure that the `CONFIG` variable in the script points to the correct configuration file for your dataset.

   ```bash
   # Edit the CONFIG variable in scripts/run_eval_{dataset}.sh to point to your configured YAML file
   # For example: CONFIG=config/eval/{dataset}_direct.yaml

   # Then run the evaluation script
   bash scripts/run_eval_{dataset}.sh
   ```

2. **Test-Time Scaling**  
   Test-time scaling involves adjusting the model parameters or configurations to improve performance during evaluation. Similar to direct evaluation, ensure the `CONFIG` variable is set correctly.

   ```bash
   # Edit the CONFIG variable in scripts/run_eval_{dataset}.sh to point to your configured YAML file
   # For example: CONFIG=config/eval/{dataset}_beam.yaml

   # Then run the test-time scaling script
   bash scripts/run_eval_{dataset}.sh
   ```

### Configuration

For both direct evaluation and test-time scaling, the configuration files should be set up similarly to those used in Stage 2. Ensure that all paths, model settings, and parameters are correctly specified in your YAML configuration files.

---

This project builds upon the LLaVA and LLaMA model families. Many thanks to the open-source community for their contributions.

---

Feel free to open an issue or submit a pull request if you have questions or suggestions!

