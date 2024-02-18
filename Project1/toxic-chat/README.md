---
license: cc-by-nc-4.0
language:
- en
size_categories:
- 10K<n<100K
config_names:
- toxicchat0124
- toxicchat1123
dataset_info:
- config_name: toxicchat0124
  features:
  - name: conv_id
    dtype: string
  - name: user_input
    dtype: string
  - name: model_output
    dtype: string
  - name: human_annotation
    dtype: bool
  - name: toxicity
    dtype: int64
  - name: jailbreaking
    dtype: int64
  - name: openai_moderation
    dtype: string
- config_name: toxicchat1123
  features:
  - name: conv_id
    dtype: string
  - name: user_input
    dtype: string
  - name: model_output
    dtype: string
  - name: human_annotation
    dtype: bool
  - name: toxicity
    dtype: int64
  - name: jailbreaking
    dtype: int64
  - name: openai_moderation
    dtype: string
configs:
- config_name: toxicchat0124
  data_files:
  - split: train
    path: data/0124/toxic-chat_annotation_train.csv
  - split: test
    path: data/0124/toxic-chat_annotation_test.csv
- config_name: toxicchat1123
  data_files:
  - split: train
    path: data/1123/toxic-chat_annotation_train.csv
  - split: test
    path: data/1123/toxic-chat_annotation_test.csv
task_categories:
- text-classification
---
## Update
[01/31/2024] We update the OpanAI Moderation API results for ToxicChat (0124) based on their updated moderation model on on Jan 25, 2024.  
[01/28/2024] We release an official [T5-Large model](https://huggingface.co/lmsys/toxicchat-t5-large-v1.0) trained on ToxicChat (toxicchat0124). Go and check it for you baseline comparision!  
[01/19/2024] We have a new version of ToxicChat (toxicchat0124)!

## Content
This dataset contains toxicity annotations on 10K user prompts collected from the Vicuna [online demo](https://chat.lmsys.org/).
We utilize a human-AI collaborative annotation framework to guarantee the quality of annotation while maintaining a feasible annotation workload.
The details of data collection, pre-processing, and annotation can be found in our [paper](https://arxiv.org/abs/2310.17389).
We believe that ToxicChat can be a valuable resource to drive further advancements toward building a safe and healthy environment for user-AI interactions.

## Version
The version name is the update time of the dataset, e.g, 0124 means it is updated on Jan, 2024. We recommend using the latest version
for training and evaluating a model. Please make sure the version of the data is the same when comparing different models. You can use the
following code to specify the dataset version:
```python
from datasets import load_dataset
dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
```
- **toxicchat0124** Based on version 1123, we did a model error analysis to check if there are any annotation errors and later fixed them. Each fix was checked by two annotators. The total label difference is 1.28% for toxicity labels and 0.34% for jailbreaking labels. We finally add 20 more human annotated examples which are not annotated in version 1123.
- **toxicchat1123:** The initial version.

**Basic Statistics**
| Version | 1123 | 0124 |
| --- | --- | --- |
| # User Prompts | 10,165 | 10,165 |
| # Human Annotations | 5,634 | 5,654 |
| % Toxic Examples | 7.18% | 7.33% |
| % Jailbreaking Examples | 1.78% | 2.01% |

## Model
We finetuned a [T5-large model](https://huggingface.co/lmsys/toxicchat-t5-large-v1.0) on ToxicChat (version 0124),
and you can use it as a baseline model for comparision. Note to have the same version of data.

| Model | Precision | Recall | F1 | AUPRC |
| --- | --- | --- | --- | --- |
| ToxicChat-T5-large | 0.7983 | 0.8475 | 0.8221 | 0.8850 |
| OpenAI Moderation (Updated Jan 25, 2024, threshold=0.02) | 0.5476 | 0.6989 | 0.6141 | 0.6313 |

## Disclaimer and Terms
- This dataset is based on the user query collected from the Vicuna online demo.
The Vicuna demo is fully anonymous for the users and also highlights the possible reuse of the user query data.
We have carefully gone through the data and taken out anything that could have personal information in it.
However, there is still a chance that some personal information might be left in the data.
If you come across anything in the data that you think should not be made public, please let us know right away.

- Safety and Moderation: **This dataset may contain racism, sexuality, or other undesired content.**
Before the annotation, the annotators are first notified about the toxic data that they will be annotated.
Verbal agreements were obtained before annotation.

- Non-Endorsement: Statements or opinions made in this dataset **do not reflect** the views of researchers or institutions involved in the data collection effort.

- Legal Compliance: Users of this data are responsible for ensuring its appropriate use.
The dataset should not be utilized for training dialogue agents, or any other applications, in manners that conflict with legal and ethical standards.

- Non-Identification: Users of this data agree to not attempt to determine the identity of individuals in this dataset.

## License
Both the user prompts and the model outputs are licensed under CC-BY-NC-4.0.

## Citation
```
@misc{lin2023toxicchat,
      title={ToxicChat: Unveiling Hidden Challenges of Toxicity Detection in Real-World User-AI Conversation}, 
      author={Zi Lin and Zihan Wang and Yongqi Tong and Yangkun Wang and Yuxin Guo and Yujia Wang and Jingbo Shang},
      year={2023},
      eprint={2310.17389},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```