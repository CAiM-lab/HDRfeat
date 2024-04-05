# HDRfeat
<small> If you use the code, please cite the following publication, where the method is described and compared with others:

[Lingkai Zhu, Fei Zhou, Bozhi Liu, Orcun Goksel: "HDRfeat: A Feature-Rich Network for High Dynamic Range Image Reconstruction" arXiv:2211.04238 (2022).](https://arxiv.org/abs/2211.04238)

## Environment
+ Python 3.8
with libraries in requirements.txt
## Data Download
Train and test data is from https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/
## Data Preprocessing
Run script under ./GenerH5Data, prepare the train data location list `train.txt` and test data location list `test.txt`
## Train 
Run `python script_training.py`
## Test
Run `python script_test.py` with `HDRfeat_model.pkl` to reproduce the results on the paper
## Result
All Reconstructed HDR images are offered at `result_hdr.tar.gz`
## Evaluation
Run `evaluate_metrics.m` to produce the metrics used in the paper
## Citation
```
@misc{zhu2022hdrfeat,
      title={HDRfeat: A Feature-Rich Network for High Dynamic Range Image Reconstruction}, 
      author={Lingkai Zhu and Fei Zhou and Bozhi Liu and Orcun Goksel},
      year={2022},
      eprint={2211.04238},
      archivePrefix={arXiv}
}
```
