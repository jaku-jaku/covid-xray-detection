import os

import time
import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum, auto
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import copy

import torch as t
import torchvision.transforms as ttf
from torchvision.datasets import MNIST
from sklearn.metrics import confusion_matrix, classification_report
import jx_lib

#######################
##### Functions #######
#######################
def get_files(DIR:str, file_end:str=".png"):
    return [ os.path.join(DIR, f) for f in os.listdir(DIR) if f.endswith(file_end) ]

def create_all_folders(DIR:str):
    path_ = ""
    for folder_name_ in DIR.split("/"):
        path_ = os.path.join(path_, folder_name_)
        create_folder(path_, False)

def clean_folder(DIR:str):
    create_folder(DIR=DIR, clean=True)

def create_folder(DIR:str, clean:bool=False):
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    elif clean:
        filelist = get_files(DIR)
        for f in filelist:
            os.remove(f)

#######################
######## ENUM #########
#######################
class VerboseLevel(IntEnum):
    NONE    = auto()
    LOW     = auto()
    MEDIUM  = auto()
    HIGH    = auto()

#############################
######## DATA CLASS #########
#############################

@dataclass
class ProgressReport:
    history: Dict

    def __init__(self):
        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "train_time": [],
            "test_loss": [],
            "test_acc": [],
            "test_time": [],
            "learning_rate": [],
        }

    
    def append(
        self,
        epoch,
        train_loss,
        train_acc,
        train_time,
        test_loss,
        test_acc,
        test_time,
        learning_rate,
    ):
        self.history["epoch"]         .append(epoch        )
        self.history["train_loss"]    .append(train_loss   )
        self.history["train_acc"]     .append(train_acc    )
        self.history["train_time"]    .append(train_time   )
        self.history["test_loss"]     .append(test_loss    )
        self.history["test_acc"]      .append(test_acc     )
        self.history["test_time"]     .append(test_time    )
        self.history["learning_rate"] .append(learning_rate)
        return ('    epoch {} > Training: [LOSS: {:.4f} | ACC: {:.4f}] | Testing: [LOSS: {:.4f} | ACC: {:.4f}] Ellapsed: {:.2f} s | rate:{:.5f}'.format(
                epoch, train_loss, train_acc, test_loss, test_acc, train_time, test_time, learning_rate
        ))


    def output_progress_plot(
        self,
        figsize       = (15,12),
        OUT_DIR       = "",
        tag           = "",
        verbose_level = VerboseLevel.MEDIUM,
    ):
        xs = self.history['epoch']
        # Plot
        fig = plt.figure(figsize=figsize)
        plt.subplot(2, 1, 1)
        plt.plot(xs, self.history['train_acc'], label="training")
        plt.plot(xs, self.history['test_acc'], label="testing")
        plt.ylabel("Accuracy")
        plt.xlabel("epoch")
        plt.xticks(xs)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(xs, self.history['train_loss'], label="training")
        plt.plot(xs, self.history['test_loss'], label="testing")
        plt.xticks(xs)
        plt.ylabel("Loss (cross-entropy)")
        plt.xlabel("epoch")
        plt.legend()

        fig.savefig("{}/training_progress[{}].png".format(OUT_DIR, tag), bbox_inches = 'tight')
        if verbose_level >= VerboseLevel.MEDIUM:
            plt.show()
        else:
            plt.close(fig)
        return fig


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., p=1):
        self.std = std
        self.mean = mean
        self.p = p

    def __call__(self, tensor):
        p = np.random.random()
        if (p <= self.p):
            return tensor + t.randn(tensor.size()) * self.std + self.mean 
        else:
            return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

################################
######## EX 1 : Helper #########
################################
class CNN_MODEL_TRAINER:        
    # # LOAD DATASET: --- ----- ----- ----- ----- ----- ----- ----- ----- #
    # # Definition:
    # @staticmethod
    # def load_mnist_data(
    #     batch_size   : int, 
    #     resize       : Optional[tuple] = None,
    #     n_workers    : int  = 1,
    #     root         : str  = "./data/",
    #     augmentation : List[str] = ["HFlip-1", "VFlip-1", "GAUSS-0.01"],
    #     shuffle      : bool = True,
    #     train_set    : bool = True,
    # ):
    #     print("=== Loading Data ... ")
    #     trans = []

    #     # Image augmentation
    #     if resize:
    #         print("> Resized to {}".format(resize))
    #         trans.append(ttf.Resize(size=resize))
    #     if augmentation is not None:
    #         print("> Augmentation: {}".format(augmentation))
    #         if "HFlip" in augmentation:
    #             trans.append(ttf.RandomHorizontalFlip())
    #         elif "HFlip-1" in augmentation:
    #             trans.append(ttf.RandomHorizontalFlip(p=1))
    #         if "VFlip" in augmentation:
    #             trans.append(ttf.RandomVerticalFlip())
    #         elif "VFlip-1" in augmentation:
    #             trans.append(ttf.RandomVerticalFlip(p=1))

    #     trans.append(ttf.ToTensor())

    #     # Gaussian Noise
    #     if augmentation is not None:
    #         # Gaussian Noise
    #         if "GAUSS-0p01" in augmentation:
    #             trans.append(AddGaussianNoise(std=0.01))
    #         elif "GAUSS-0p1" in augmentation:
    #             trans.append(AddGaussianNoise(std=0.1))
    #         elif "GAUSS-1" in augmentation:
    #             trans.append(AddGaussianNoise(std=1))
    #         elif "GAUSS-0p5-0p5" in augmentation:
    #             trans.append(AddGaussianNoise(std=0.5, p=0.5))

    #     transform = ttf.Compose(trans)
        
    #     data = MNIST(root=root, train=train_set, download=True, transform=transform)
    #     dataset = t.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers)

    #     print("=== Data Loaded [{}] ===".format("Testing" if train_set else "Training"))
    #     return dataset

    # TESTING:  ----- ----- ----- ----- ----- ----- ----- ----- ----- #
    @staticmethod
    def test(
        device,
        test_dataset, 
        net, 
        loss_func,
        max_data_samples: Optional[int] = None,
        verbose_level: VerboseLevel = VerboseLevel.LOW,
        _print=print,
    ):
        y_true = []
        y_pred = []
        
        if verbose_level >= VerboseLevel.LOW:
            _print("  >> Testing (wip) ")

        test_loss_sum, test_acc_sum, test_n, test_start = 0.0, 0.0, 0, time.time()

        batch_count = 0
        for i, (X, y) in enumerate(test_dataset):
            if max_data_samples is not None:
                if i >= max_data_samples:
                    break
                if verbose_level >= VerboseLevel.HIGH:
                    print("\r   >[{}/{}]".format(i+1, max_data_samples),  end='')
            elif verbose_level >= VerboseLevel.HIGH:
                print("\r   >[{}/{}]".format(i+1, len(test_dataset)),  end='')
            
            # hardware-acceleration
            if device != None:
                X = X.to(device)
                y = y.to(device)

            # Predict:
            y_prediction = net(X)
            # Calculate loss
            loss = loss_func(y_prediction, y)
            # Compute Accuracy
            test_loss_sum += loss.item()
            test_acc_sum += (y_prediction.argmax(dim=1) == y).sum().item()
            test_n += y.shape[0]
            batch_count += 1

            # accumulate:
            y_true.extend(y.cpu().detach().numpy())
            y_pred.extend(y_prediction.argmax(dim=1).cpu().detach().numpy())

        print("\r")
        test_loss = test_loss_sum / batch_count
        test_acc = test_acc_sum / test_n
        test_ellapse = time.time() - test_start

        return test_loss, test_acc, test_n, test_ellapse, y_true, y_pred

    # TRAINING: ----- ----- ----- ----- ----- ----- ----- ----- ----- #
    @staticmethod
    def train(
        device,
        train_dataset, 
        net, 
        optimizer, 
        loss_func,
        max_data_samples: Optional[int] = None,
        verbose_level: VerboseLevel = VerboseLevel.LOW,
        _print=print,
    ):
        # Training:
        if verbose_level >= VerboseLevel.LOW:
            _print("  >> Learning (wip) ")
        train_loss_sum, train_acc_sum, train_n, train_start = 0.0, 0.0, 0, time.time()
        batch_count = 0
        for i, (X, y) in enumerate(train_dataset):
            if max_data_samples is not None:
                if i >= max_data_samples:
                    break
                if verbose_level >= VerboseLevel.HIGH:
                    print("\r   >[{}/{}]".format(i+1, max_data_samples), end='')
            elif verbose_level >= VerboseLevel.HIGH:
                print("\r   >[{}/{}]".format(i+1, len(train_dataset)),  end='')
            
            # 1: Clear PyTorch Cache
            optimizer.zero_grad()
            
            # 2: hardware-acceleration
            if device != None:
                X = X.to(device)
                y = y.to(device)

            # 3: FWD:
            y_prediction = net(X)

            # 4: backpropagation (training)
            loss = loss_func(y_prediction, y)
            loss.backward()
            optimizer.step()
            # Compute Accuracy
            train_loss_sum += loss.item()
            train_acc_sum += (y_prediction.argmax(dim=1) == y).sum().item()
            train_n += y.shape[0]
            batch_count += 1
        
        print("\r")
        train_loss = train_loss_sum / batch_count
        train_acc = train_acc_sum / train_n
        train_ellapse = time.time() - train_start

        return train_loss, train_acc, train_n, train_ellapse

    @staticmethod
    def train_and_monitor(
        device,
        train_dataset, 
        test_dataset, 
        optimizer, 
        loss_func,
        net, 
        model_output_path,
        target_names,
        num_epochs: int,
        early_stopping_n_epochs_consecutive_decline: int = 3,
        # history_epoch_resolution: float = 1.0, TODO: mini-batches progress!!!
        max_data_samples: Optional[int] = None,
        verbose_level: VerboseLevel = VerboseLevel.LOW,
        _print=print,
    ):
        report = ProgressReport()
        n_decline = 0
        best_test_acc = 0
        best_net = None
        t_start = time.time()
        # Cross entropy
        for epoch in range(num_epochs):
            if verbose_level >= VerboseLevel.LOW:
                _print("> epoch {}/{}:".format(epoch + 1, num_epochs))
            
            # Train:
            train_loss, train_acc, train_n, train_ellapse = CNN_MODEL_TRAINER.train(
                device = device,
                train_dataset = train_dataset, 
                net = net, 
                optimizer = optimizer, 
                loss_func = loss_func,
                max_data_samples = max_data_samples,
                verbose_level = verbose_level,
                _print= _print,
            )
            
            # Testing:
            test_loss, test_acc, test_n, test_ellapse, y_true, y_pred = CNN_MODEL_TRAINER.test(
                device = device,
                test_dataset = test_dataset, 
                net = net, 
                loss_func = loss_func,
                max_data_samples = max_data_samples,
                verbose_level = verbose_level,
                _print= _print,
            )

            # Store
            _print("{}\n".format(report.append(
                epoch         = epoch + 1,
                train_loss    = train_loss,
                train_acc     = train_acc,
                train_time    = train_ellapse,
                test_loss     = test_loss,
                test_acc      = test_acc,
                test_time     = test_ellapse,
                learning_rate = optimizer.param_groups[0]["lr"],
            )))

            if test_acc > best_test_acc:
                best_net = copy.deepcopy(net)
                best_test_acc = test_acc
                path = "{}/best_state_dict_{}:{}.pth".format(model_output_path, epoch + 1, num_epochs)
                t.save(net.state_dict(), path)
                _print("> Found Best Model State Dict saved @{} \n".format(path))
                # generate evaluation report:

                #  Generate Evaluation Report:
                cm = confusion_matrix(y_true, y_pred )
                clr = classification_report(y_true, y_pred, target_names=target_names)

                # Output:
                fig, status = jx_lib.make_confusion_matrix(cf=cm)
                fig.savefig("{}/confusion_matrix_{}:{}.jpg".format(model_output_path, epoch + 1, num_epochs), bbox_inches = 'tight')
                print("Best Classification Report:\n----------------------\n", clr)

            else:
                n_decline += 1
            
            # early stopping:
            if n_decline > early_stopping_n_epochs_consecutive_decline:
                break

        # Store the best model:
        t.save(best_net, "{}/best_model_{}.pth \n".format(model_output_path, epoch + 1))

        _print("=> TOTAL Training + Validation Time: {:.3f} min".format((time.time() - t_start)/60))
        return report, best_net
