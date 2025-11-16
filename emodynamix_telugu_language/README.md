<p align="center">
  <h1 align="center"><img src="img/logo.png" alt="Logo" style="height:1em; vertical-align:middle;"> EmoDynamiX</h1>
  <h3 align="center">Emotional Support Dialogue Strategy Prediction by Modelling MiXed Emotions and Discourse Dynamics</h3>
  <h4 align="center"><img src="img/acl-logo.png" alt="ACL Logo" style="height:1em; vertical-align:middle;"> <i>NAACL 2025 Oral</i></h4>
  <p align="center">  
    <a href="https://arxiv.org/pdf/2408.08782">Paper</a>
    ·
    <a href="https://github.com/cw-wan/EmoDynamiX-v2/blob/master/Slides.pdf">Slides</a>
    ·
    <a href="https://github.com/cw-wan/EmoDynamiX-v2/blob/master/Poster.pdf">Poster*</a>
  </p>
  <span style="font-size: 60%;">* <i>Poster presented at <a href="https://coria-taln-2025.lis-lab.fr/">CORIA-TALN 2025</a>.</i></span>
</p>

![](img/architecture.jpg)

## Usage

[DEMO.ipynb](DEMO.ipynb) shows an example of using EmoDynamiX. You could integrate EmoDynamiX with any LLM you like to make your own strategy-controlled ESC agent!

## Download Checkpoints

|              Model               |                                                                                        URL                                                                                        |
|:--------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|       EmoDynamiX on ESConv       |                                           [link](https://drive.google.com/file/d/1pbBH5pbw5bY-35avobkdzqi0gv_bL_pn/view?usp=drive_link)                                           |
|       EmoDynamiX on AnnoMI       |                                           [link](https://drive.google.com/file/d/1VWhx9xoC7L9roSPeP9hvXjGlyjzs-kY5/view?usp=drive_link)                                           |
| Pretrained Submodules | [link](https://drive.google.com/file/d/1KNsoWp1FjdMnrCVWiONRb6w4QUpzGuyP/view?usp=drive_link) |

Unzip to the project root directory.

## Reproduce the Results

Test on ESConv:

```shell
./test_roberta_hg_esconv.sh
```

Test on AnnoMI:

```shell
./test_roberta_hg_annomi.sh
```

## Training from Scratch

Train on ESConv:

```shell
./train_roberta_hg_esconv.sh
```

Train on AnnoMI:

```shell
./train_roberta_hg_annomi.sh
```

## Telugu Adaptation (XLM-R Large)

Train on translated ESConv Telugu:

```powershell
./train_telugu.ps1 -Model xlmr-hg-telugu -Dataset esconv-telugu -BatchSize 8 -Epochs 10 -TotalSteps 5000 -HgDim 512 -TeluguErcPath telugu_erc_xlmroberta_trained_v2
```

Preprocess for lightmode (fast graph training):

```powershell
python make_telugu_preprocessed.py --telugu_erc_path telugu_erc_xlmroberta_trained_v2
```

Then train with parser disabled and cached logits:

```powershell
python main.py --mode train --model xlmr-hg-telugu --dataset esconv-telugu-preprocessed --lightmode 1 --total_steps 5000 --total_epochs 10 --batch_size 8 --hg_dim 512 --telugu_erc_path telugu_erc_xlmroberta_trained_v2
```

Evaluate a saved checkpoint (replace STEP):

```powershell
./infer_telugu.ps1 -CheckpointStep STEP -Model xlmr-hg-telugu -Dataset esconv-telugu -BatchSize 8 -TeluguErcPath telugu_erc_xlmroberta_trained_v2
```

Precompute ERC logits externally and pass via `samples['erc_logits']` or graph-space embeddings via `samples['erc_embeddings']` for faster training.

## Citation

If you find our work useful, please cite our paper:

```bibtex
@inproceedings{wan-etal-2025-emodynamix,
  author       = {Chenwei Wan and
                  Matthieu Labeau and
                  Chlo{\'{e}} Clavel},
  editor       = {Luis Chiruzzo and
                  Alan Ritter and
                  Lu Wang},
  title        = {EmoDynamiX: Emotional Support Dialogue Strategy Prediction by Modelling
                  MiXed Emotions and Discourse Dynamics},
  booktitle    = {Proceedings of the 2025 Conference of the Nations of the Americas
                  Chapter of the Association for Computational Linguistics: Human Language
                  Technologies, {NAACL} 2025 - Volume 1: Long Papers, Albuquerque, New
                  Mexico, USA, April 29 - May 4, 2025},
  pages        = {1678--1695},
  publisher    = {Association for Computational Linguistics},
  year         = {2025},
  url          = {https://doi.org/10.18653/v1/2025.naacl-long.81},
  doi          = {10.18653/V1/2025.NAACL-LONG.81},
  timestamp    = {Fri, 13 Jun 2025 08:28:21 +0200},
  biburl       = {https://dblp.org/rec/conf/naacl/WanLC25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
