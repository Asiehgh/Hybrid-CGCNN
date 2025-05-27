import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
import torch.distributions as dist
from cgcnn.data import CIFData
from cgcnn.data import collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks - Improved')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--atom-fea-len', default=64, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')
                    
parser.add_argument('--feature-file', type=str, required=True,
                    help='Path to the file containing 5-ring and HOMO-LUMO data')
                    
parser.add_argument('--patience', default=25, type=int,
                    help='number of epochs to wait before early stopping (default: 25)')
parser.add_argument('--min-delta', default=0.0005, type=float,
                    help='minimum change in validation loss to qualify as improvement (default: 0.0005)')

parser.add_argument('--warmup-epochs', default=8, type=int,
                    help='number of warmup epochs (default: 8)')
parser.add_argument('--use-plateau-scheduler', action='store_true',
                    help='use ReduceLROnPlateau instead of MultiStepLR')
parser.add_argument('--gradient-clip-norm', default=0.5, type=float,
                    help='gradient clipping max norm (default: 0.5)')
parser.add_argument('--stability-check-window', default=5, type=int,
                    help='window size for stability checking (default: 5)')

args = parser.parse_args(sys.argv[1:])
args.cuda = not args.disable_cuda and torch.cuda.is_available()

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.

class ImprovedEarlyStopping:
    """Enhanced early stopping with stability monitoring"""
    def __init__(self, patience=20, min_delta=0.001, restore_best_weights=True, 
                 stability_window=5):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.stability_window = stability_window
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.val_history = []
        
    def __call__(self, val_loss, model=None):
        self.val_history.append(val_loss)
        
        if len(self.val_history) > 20:
            self.val_history = self.val_history[-20:]
        
        if self.best_score is None:
            self.best_score = val_loss
            if model is not None and self.restore_best_weights:
                self.best_model_state = model.state_dict().copy()
        elif val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_model_state = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
    def get_stability_metrics(self):
        """Calculate stability metrics for recent validation performance"""
        if len(self.val_history) < self.stability_window:
            return None, None
            
        recent_vals = self.val_history[-self.stability_window:]
        std_dev = np.std(recent_vals)
        trend = np.polyfit(range(len(recent_vals)), recent_vals, 1)[0]
        
        return std_dev, trend
    
    def restore_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)

class ValidationStabilityMonitor:
    """Monitor validation stability and provide feedback"""
    def __init__(self, window_size=5, stability_threshold=0.3):
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.history = []
        
    def update(self, val_loss):
        self.history.append(val_loss)
        if len(self.history) > self.window_size:
            self.history.pop(0)
    
    def is_stable(self):
        if len(self.history) < self.window_size:
            return True
        return np.std(self.history) <= self.stability_threshold
    
    def get_stability_score(self):
        if len(self.history) < 2:
            return 1.0
        return 1.0 / (1.0 + np.std(self.history))

def main():
    global args, best_mae_error

    data_path = os.path.abspath(args.data_options[0])
    id_prop_path = os.path.join(data_path, 'id_prop.csv')
    print(f"Current working directory: {os.getcwd()}")
    print(f"Data path argument: {args.data_options[0]}")
    print(f"Absolute data path: {data_path}")
    print(f"Looking for id_prop.csv at: {id_prop_path}")
    print(f"File exists: {os.path.exists(id_prop_path)}")

    dataset = CIFData(root_dir=args.data_options[0], feature_file=args.feature_file, max_num_nbr=12, radius=8)

    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        return_test=True)

    if args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        if len(dataset) < 500:
            warnings.warn('Dataset has less than 500 data points. '
                          'Lower accuracy is expected. ')
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in
                                sample(range(len(dataset)), 500)]
        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)

    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    n_extra_features = len(dataset.feature_names)
    
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                               atom_fea_len=args.atom_fea_len,
                               n_conv=args.n_conv,
                               h_fea_len=args.h_fea_len,
                               n_h=args.n_h,
                               classification=True if args.task == 'classification' else False,
                               n_extra_features=n_extra_features)

    if args.cuda:
        model.cuda()

    print("\nModel Architecture:")
    print(model)
    print(f"\nFeature Dimensions:")
    print(f"Original atom features: {orig_atom_fea_len}")
    print(f"Bond features: {nbr_fea_len}")
    print(f"Extra features: {n_extra_features} features")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay,
                               eps=1e-8, amsgrad=True)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, weights_only=False)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    early_stopping = ImprovedEarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        restore_best_weights=True,
        stability_window=args.stability_check_window
    )
    
    stability_monitor = ValidationStabilityMonitor(
        window_size=args.stability_check_window,
        stability_threshold=0.3
    )

    if args.use_plateau_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                     patience=5, verbose=True, min_lr=1e-6)
    else:
        scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

    train_losses = []
    val_losses = []
    learning_rates = []
    stability_scores = []

    print(f"\n{'='*70}")
    print(f"IMPROVED TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Optimizer: {args.optim} (with AMSGrad for Adam)")
    print(f"Initial Learning Rate: {args.lr}")
    print(f"Scheduler: {'ReduceLROnPlateau' if args.use_plateau_scheduler else 'MultiStepLR'}")
    if not args.use_plateau_scheduler:
        print(f"LR Milestones: {args.lr_milestones}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Epochs: {args.epochs}")
    print(f"Warmup Epochs: {args.warmup_epochs}")
    print(f"Early Stopping - Patience: {args.patience}, Min Delta: {args.min_delta}")
    print(f"Gradient Clipping: Max Norm {args.gradient_clip_norm}")
    print(f"Stability Monitoring: Window size {args.stability_check_window}")
    print(f"Dataset Size: {len(dataset)} samples")
    print(f"Train/Val/Test Split: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")
    print(f"{'='*70}\n")

    print(f"Starting improved training...")
    training_start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()

        if epoch < args.warmup_epochs:
            lr_scale = (epoch + 1) / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * lr_scale
            print(f"Warmup phase: Epoch {epoch+1}/{args.warmup_epochs}, LR scale: {lr_scale:.3f}")

        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        print(f"\nEpoch [{epoch+1}/{args.epochs}] - LR: {current_lr:.6f}")
        print("-" * 60)

        epoch_train_loss = train(train_loader, model, criterion, optimizer, epoch, normalizer, args)

        print("Validating...")
        mae_error = validate(val_loader, model, criterion, normalizer, args)

        if mae_error != mae_error or mae_error == float('inf'):
            print('CRITICAL ERROR: NaN/Inf in validation! Stopping training.')
            sys.exit(1)

        stability_monitor.update(mae_error)
        stability_score = stability_monitor.get_stability_score()
        stability_scores.append(stability_score)

        if epoch_train_loss is not None:
            train_losses.append(epoch_train_loss)
        val_losses.append(mae_error)

        if epoch >= args.warmup_epochs:
            if args.use_plateau_scheduler:
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(mae_error)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != old_lr:
                    print(f" ReduceLROnPlateau: {old_lr:.6f} → {new_lr:.6f}")
            else:
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step()
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != old_lr:
                    print(f" MultiStepLR: {old_lr:.6f} → {new_lr:.6f}")

        if args.task == 'regression':
            is_best = mae_error < best_mae_error
            if is_best:
                best_mae_error = mae_error
        else:
            is_best = mae_error > best_mae_error
            if is_best:
                best_mae_error = mae_error

        std_dev, trend = early_stopping.get_stability_metrics()
        if std_dev is not None:
            stability_status = " Stable" if stability_monitor.is_stable() else " Unstable"
            print(f" Validation Stability: {stability_status}")
            print(f"   Recent {args.stability_check_window}-epoch std: {std_dev:.4f}")
            print(f"   Trend: {' Rising' if trend > 0.01 else ' Falling' if trend < -0.01 else ' Stable'}")
            print(f"   Stability Score: {stability_score:.3f}")

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'learning_rates': learning_rates,
            'stability_scores': stability_scores
        }, is_best, filename='checkpoint_improved.pth.tar')

        if early_stopping(mae_error, model):
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - training_start_time

            print(f"\n{'='*70}")
            print(f" EARLY STOPPING TRIGGERED")
            print(f"{'='*70}")
            print(f"Epoch: {epoch+1}/{args.epochs}")
            print(f"Best Validation MAE: {early_stopping.best_score:.4f}")
            print(f"Current Validation MAE: {mae_error:.4f}")
            print(f"Patience exhausted: {early_stopping.counter}/{early_stopping.patience}")
            print(f"Training time: {total_time/60:.1f} minutes")

            if early_stopping.restore_best_weights and early_stopping.best_model_state is not None:
                print(" Restoring best model weights...")
                early_stopping.restore_best_model(model)

            print(f"{'='*70}\n")
            break

        epoch_time = time.time() - epoch_start_time
        if early_stopping.counter > 0:
            patience_bar = "█" * early_stopping.counter + "░" * (early_stopping.patience - early_stopping.counter)
            print(f" EarlyStopping: [{patience_bar}] {early_stopping.counter}/{early_stopping.patience}")
            print(f"   Best: {early_stopping.best_score:.4f}")
        else:
            if is_best:
                print(f" New best validation MAE: {mae_error:.4f}")

        print(f"  Epoch time: {epoch_time:.1f}s")

        if args.cuda:
            torch.cuda.empty_cache()

    total_training_time = time.time() - training_start_time
    final_epoch = min(epoch + 1, args.epochs)

    print(f"\n{'='*70}")
    print(f" IMPROVED TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Total epochs: {final_epoch}")
    print(f"Total training time: {total_training_time/60:.1f} minutes")
    print(f"Average time per epoch: {total_training_time/final_epoch:.1f}s")
    print(f"Best validation MAE: {best_mae_error:.4f}")
    
    if len(stability_scores) > 5:
        avg_stability = np.mean(stability_scores[-10:])
        print(f"Final stability score: {avg_stability:.3f}")

    print(f"{'='*70}\n")

    print('-' * 60)
    print(' EVALUATING BEST MODEL ON TEST SET')
    print('-' * 60)

    if os.path.exists('model_best.pth.tar'):
        print(" Loading best model checkpoint...")
        best_checkpoint = torch.load('model_best.pth.tar', , weights_only=False)
        model.load_state_dict(best_checkpoint['state_dict'])
    else:
        print("  Best model checkpoint not found, using current model state")

    test_mae = validate(test_loader, model, criterion, normalizer, args, test=True)

    print(f"\n{'='*70}")
    print(f" FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Best Validation MAE: {best_mae_error:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    val_test_diff = abs(best_mae_error - test_mae)
    if val_test_diff < 0.05:
        print(" Excellent generalization")
    elif val_test_diff < 0.15:
        print(" Good generalization")
    elif val_test_diff < 0.3:
        print("  Moderate generalization gap")
    else:
        print(" Large generalization gap - possible overfitting")

    print(f"Validation-Test difference: {val_test_diff:.4f}")
    print(f"{'='*70}\n")

    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'stability_scores': stability_scores,
        'best_val_mae': best_mae_error,
        'test_mae': test_mae,
        'total_epochs': final_epoch,
        'training_time': total_training_time,
        'final_stability': np.mean(stability_scores[-5:]) if len(stability_scores) >= 5 else 0
    }

    try:
        import pickle
        with open('training_history_improved.pkl', 'wb') as f:
            pickle.dump(training_history, f)
        print(" Enhanced training history saved to 'training_history_improved.pkl'")
    except Exception as e:
        print(f" Could not save training history: {e}")

    return model, val_loader

def train(train_loader, model, criterion, optimizer, epoch, normalizer, args):
    """Enhanced training function with improved monitoring"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    model.train()
    
    end = time.time()
    epoch_losses = []
    gradient_norms = []
    
    for i, (input, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.cuda:
           input_var = (Variable(input[0].cuda(non_blocking=True)),
                       Variable(input[1].cuda(non_blocking=True)),
                       input[2].cuda(non_blocking=True),
                       [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
                       Variable(input[4].cuda(non_blocking=True)))
        else:
           input_var = (Variable(input[0]),
                       Variable(input[1]),
                       input[2],
                       input[3],
                       Variable(input[4]))
        
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
            
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        output = model(*input_var)
        loss = criterion(output, target_var)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  Warning: NaN/Inf loss in training batch {i}, skipping...")
            continue
            
        epoch_losses.append(loss.item())

        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        optimizer.zero_grad()
        loss.backward()

        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** (1. / 2)
        gradient_norms.append(total_grad_norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip_norm)
        if total_grad_norm > 5.0:
            print(f"  Warning: Large gradient norm ({total_grad_norm:.2f}) in epoch {epoch}, batch {i}")
        
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            if args.task == 'regression':
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t'
                      'GradNorm {grad_norm:.2f}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, mae_errors=mae_errors,
                    grad_norm=total_grad_norm))
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})\t'
                      'GradNorm {grad_norm:.2f}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, accu=accuracies,
                    prec=precisions, recall=recalls, f1=fscores,
                    auc=auc_scores, grad_norm=total_grad_norm))

    if gradient_norms:
        avg_grad_norm = np.mean(gradient_norms)
        max_grad_norm = np.max(gradient_norms)
        print(f" Gradient Stats - Avg: {avg_grad_norm:.3f}, Max: {max_grad_norm:.3f}")
        
        if avg_grad_norm > 2.0:
            print("  High average gradient norm detected - consider reducing learning rate")
    
    return np.mean(epoch_losses) if epoch_losses else None


def validate(val_loader, model, criterion, normalizer, args, test=False):
    """Enhanced validation function with improved error handling"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    
    model.eval()
    problematic_batches = 0
    
    dataset_type = "Test" if test else "Validation"
    
    with torch.no_grad():
        for i, (input, target, batch_cif_ids) in enumerate(val_loader):
            try:

                if torch.isnan(input[0]).any() or torch.isinf(input[0]).any():
                    print(f"  Warning: NaN/Inf in batch {i} atom features")
                    problematic_batches += 1
                    continue
                    
                if torch.isnan(input[4]).any() or torch.isinf(input[4]).any():
                    print(f"  Warning: NaN/Inf in batch {i} extra features")
                    problematic_batches += 1
                    continue
                
                if torch.isnan(target).any() or torch.isinf(target).any():
                    print(f"  Warning: NaN/Inf in batch {i} targets")
                    problematic_batches += 1
                    continue
                
                if args.cuda:
                    input_var = (Variable(input[0].cuda(non_blocking=True)),
                                Variable(input[1].cuda(non_blocking=True)),
                                input[2].cuda(non_blocking=True),
                                [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
                                Variable(input[4].cuda(non_blocking=True)))
                else:
                    input_var = (Variable(input[0]),
                                Variable(input[1]),
                                input[2],
                                input[3],
                                Variable(input[4]))
                

                target_normed = normalizer.norm(target)
                if args.cuda:
                    target_var = Variable(target_normed.cuda(non_blocking=True))
                else:
                    target_var = Variable(target_normed)
                

                output = model(*input_var)
                

                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"  Warning: NaN/Inf in model output for batch {i}")
                    problematic_batches += 1
                    continue
                
                loss = criterion(output, target_var)
                

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  Warning: NaN/Inf loss in batch {i}")
                    problematic_batches += 1
                    continue
                
                mae_error = mae(normalizer.denorm(output.data.cpu()), target)
                
                if mae_error < 50:
                    losses.update(loss.data.cpu().item(), target.size(0))
                    mae_errors.update(mae_error, target.size(0))
                else:
                    print(f"  Warning: Extremely high MAE ({mae_error:.2f}) in batch {i}")
                    problematic_batches += 1
                
            except Exception as e:
                print(f" Error in {dataset_type.lower()} batch {i}: {e}")
                problematic_batches += 1
                continue
    
    if problematic_batches > 0:
        print(f"  Found {problematic_batches} problematic {dataset_type.lower()} batches")
    
    if mae_errors.avg > 5.0:
        print(f"  WARNING: Very high {dataset_type.lower()} MAE ({mae_errors.avg:.3f}) - possible training instability")
    elif mae_errors.avg < 0.5:
        print(f" Excellent {dataset_type.lower()} performance: MAE {mae_errors.avg:.3f}")
    
    print(f' {dataset_type} MAE: {mae_errors.avg:.4f}')
    return mae_errors.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        

if __name__ == '__main__':
    model, val_loader = main()
