CGCNN-Plus: Enhanced Crystal Graph Convolutional Neural Network

A hybrid machine learning framework that combines crystal graph representations with molecular-level descriptors for improved materials property prediction. Built upon the original CGCNN architecture with significant enhancements for training stability and predictive performance.


Installation:
conda create -n cgcnn-plus python=3.8+

conda activate cgcnn-plus

conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

pip install pymatgen scikit-learn numpy



Usage:
python main.py dataset_path --feature-file features.csv

Training Parameters
Required

dataset_path: Path to your dataset directory
--feature-file: Path to molecular descriptors CSV file

Optional Training

--epochs: Number of training epochs (default: 30)
--batch-size: Batch size (default: 256)
--lr: Initial learning rate (default: 0.01)
--optim: Optimizer [SGD, Adam] (default: SGD)
--patience: Early stopping patience (default: 25)
--min-delta: Minimum improvement threshold (default: 0.0005)

Advanced Options

--use-plateau-scheduler: Use ReduceLROnPlateau instead of MultiStepLR
--warmup-epochs: Number of warmup epochs (default: 8)
--gradient-clip-norm: Gradient clipping threshold (default: 0.5)
--stability-check-window: Window for stability monitoring (default: 5)

Model Architecture

--atom-fea-len: Atom feature length (default: 64)
--h-fea-len: Hidden feature length (default: 128)
--n-conv: Number of conv layers (default: 3)
--n-h: Number of hidden layers (default: 1)

After training, CGCNN-Plus generates:
model_best.pth.tar: Best model checkpoint
checkpoint_improved.pth.tar: Latest checkpoint with full training state
test_results.csv: Test set predictions
training_history_improved.pkl: Complete training metrics





