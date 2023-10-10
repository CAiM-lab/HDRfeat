# HDRfeat
<small> If you use the code, please cite the following publication, where the method is described and compared with others:

[Lingkai Zhu, Fei Zhou, Bozhi Liu, Orcun Goksel: "HDRfeat: A Feature-Rich Network for High Dynamic Range Image Reconstruction" arXiv:2211.04238 (2022).](https://arxiv.org/abs/2211.04238)

## environment
+ Python 3.8
with libraries in requirements.txt
## Data Download
Data is from https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/
## Data Preprocessing
run script under ./GenerH5Data, prepare the train data list train.txt and test data list test.txt
## Train 
run script_training.py
## Test 
run script_test.py with HDRfeat_model.pkl to reproduce the results on the paper
## Citation
@misc{zhu2022hdrfeat,
      title={HDRfeat: A Feature-Rich Network for High Dynamic Range Image Reconstruction}, 
      author={Lingkai Zhu and Fei Zhou and Bozhi Liu and Orcun Goksel},
      year={2022},
      eprint={2211.04238},
      archivePrefix={arXiv}
}
