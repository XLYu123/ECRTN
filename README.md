# ECRTN
Code accompanying the paper " Few-shot fault diagnosis of autonomous underwater vehicle based on dual-loss nonlinear independent component estimation model with channel attention residual transfer learning" by Wenliao Du, Xinlong Yu, Zhen Guo, Hongchao Wang, Ziqiang Pu, Chuan Li (Ready to be submitted for publication).

 Tensorflow 2.0 implementation
 # Requirements
- python 3.11
- Tensorflow == 2.6.2
- Numpy == 1.19.2
- Keras == 2.6.0
- Note: All experiment were excecuted with NVIDIA GeForce GTX 1050Ti

# File discription
ECRTN: The ECRTN model is proposed in this paper.
SEResNet : a squeezed excitation residual network that solves the problem of gradient vanishing in deep network training by introducing residual connections.
CGRU : an adaptive network constructed based on gated recurrent units and combining convolutional operations and global recurrent learning strategy (CGRU).
WDCNN-GRU : A two-channel multiscale parallel WDCNN-GRU-based network for solving the sample-less problem.
# Implementation details
- Hyperparameter settings: Adam optimizer is used with learning rate of in both the generator and the discriminator;The batch size is , total iteration is 10,000. LABDA (Weight of cycle consistency loss) is . Random projection in SWD is .1e-4 32 10 32
# Usage
Note: Due to the copyright, no any data set is uploaded. For more detail pelase contact Authors.
# Ackonwledgements
This work was supported in part by the National Nature Science Foundation of China (52275138, 52175112), the Key R&D Projects in Henan Province (231111221100, 241111222400),  the Program for Innovative Research Team (in Science and Technology) in University of Henan Province (25IRTSTHN024), and the State Key Laboratory of Mechanical System and Vibration (Grant no. MSV202502).
