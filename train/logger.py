import os
import sys
import time
import torch
import logging
import matplotlib.pyplot as plt

train_root = os.path.dirname(os.path.abspath(__file__))
log_root = os.path.join(train_root, "log")
if not os.path.exists(log_root):
    os.mkdir(log_root)

class Logger():
    def __init__(self, name, model, verbosity=1) -> None:
        self.name = name
        self.log_root = os.path.join(log_root, name, f"{int(time.time())}")

        self.model = model
        self.runtime_data = {
            "train_losses": [],
            "val_losses": [],
            "test_losses": [],
            "train_accs": [],
            "val_accs": [],
            "test_accs": [],
        }
        self.__init_logger(verbosity)

    def __init_logger(self, verbosity):
        level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
        formatter = logging.Formatter(
            "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
        )
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(level_dict[verbosity])

        logfile = os.path.join(self.log_root, "log")
        fh = logging.FileHandler(logfile, "w")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

    # save one indirecting
    def debug(self, *args, **kwargs):
        self.logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        self.logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        self.logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        self.logger.error(*args, **kwargs)

    def critical(self, *args, **kwargs):
        self.logger.critical(*args, **kwargs)

    # append runtime data
    def append_train_loss(self, loss):
        self.runtime_data["train_loss"].append(loss)

    def append_val_loss(self, loss):
        self.runtime_data["val_loss"].append(loss)

    def append_test_loss(self, loss):
        self.runtime_data["test_loss"].append(loss)

    def append_train_acc(self, acc):
        self.runtime_data["train_loss"].append(acc)

    def append_val_acc(self, acc):
        self.runtime_data["val_loss"].append(acc)
        
    def append_test_acc(self, acc):
        self.runtime_data["test_loss"].append(acc)

    # dumping the result as files
    def dump_model(self):
        model_path = os.path.join(self.log_root, "model.pth")
        torch.save(self.model, model_path)

    def plot_curve(self, name):
        assert name in self.runtime_data.keys()
        ls = self.runtime_data[name]
        plt.plot(ls)
        figfile = os.path.join(self.log_root, f"{name}.png")
        plt.savefig(figfile)
