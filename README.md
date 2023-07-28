# MMBERT

Source codes for paper:
MMBERT: A Unified Framework for Biomedical Named Entity Recognition 

1. Environments
- python (3.9.16)
- cuda (11.6)
- prettytable (2.4.0)
- numpy (1.21.4)
- torch (1.12.0)
- gensim (4.1.2)
- transformers (4.13.0)
- pandas (1.3.4)
- scikit-learn (1.0.1)
2. Input format:
====== CoNLL format (prefer BIOES tag scheme), with each character its label for one line. Sentences are splited with a null line.

胸 B-SYMPTOM 闷 E-SYMPTOM 呼 B-SYMPTOM 吸 M-SYMPTOM 困 M-SYMPTOM 难 E-SYMPTOM 是 O 怎 O 么 O 回 O 事 O 阿 O

3. Preparation
Download dataset
Process them to fit the same format as the example in data/
Put the processed data into the directory data/
##4. Pretrained Embeddings:
Pre trained model ernie health: (https://pan.baidu.com/s/14r0xQQBq2Wpo55penLj4sA, code:提取码：zdlf)

After downloading, please place it in the cache folder for replacement.

Character embeddings: [gigaword_chn.all.a2b.uni.ite50.vec](链接：https://pan.baidu.com/s/1GYsAESEvq7LnPC4uar6ezg, code：iu89)

Word embeddings: [ctb.50d.vec](链接：https://pan.baidu.com/s/194TTTKaUTbvpI8SDa-r6Pg, code：39l8)

5. Training/Deving/Testing
python main.py

6. Decoding
python evel.py

7. Citation
If you use this work or code, please kindly cite this paper:

{ title={MMBERT: A Unified Framework for Biomedical Named Entity Recognition}, author={Lei Fu，and Zuquan Wengan，Jiheng Zhang，Haihe Xie，Yiqing Cao，Chenhua Zhang }, year={2023} }
