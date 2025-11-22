# Task-Prompt

We propose multi-source task-conditional discrimination pretraining foundation model for rumor detection. 
Our core design first addresses the objective mismatch by parameterizing task semantics as a learnable task prompt and unifying both pretraining and fine-tuning into an identical binary discrimination task. 
Second, to address multi-source decision diversity, we leverage this task prompt to condition the discriminator. 
A shared task prompt conditions the discriminator, forcing it to learn a unified and learnable decision standard at the decision layer. 
Specifically, two augmentations of each claim are encoded to yield positive and negative representation pairs.
A discriminator conditioned on a shared pretraining task prompt performs binary classification of these pairs. 
For downstream fine-tuning, the encoder and discriminator are kept frozen, and the pretraining task prompt is replaced with a downstream task prompt. 
Only this downstream task prompt is updated with a few labeled samples to align the training objection.

## Dependencies

```python
pip install -r requirements.txt
```

## Dataset
The Raw Pheme dataset can be obtained from https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078.

The raw Weibo dataset can be downloaded from https://github.com/majingCUHK/Rumor_GAN.

The Politifact, Gossipcop dataset can be auto download bt Pytorch_Geometric.

The WeiboCOVID19, TwitterCOVID19 dataset can be download from https://drive.google.com/drive/folders/1gvuSeorLAljGZaD7gyWrUA0gyotT_rl6?usp=sharing.

The DRWeiboV3 dataset can be download from https://github.com/CcQunResearch/DRWeibo.

## Usage

You can use the following command, and the parameters are given

```python
python main.py --dataset DRWeiboV3
```

The `--dataset` argument should be one of [DRWeiboV3, Weibo, WeiboCOVID19, PHEME, Politifact, Gossipcop, TwitterCOVID19].
