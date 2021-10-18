
import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter



def _resume_checkpoint(self, resume_path):
    """
    Resume from saved checkpoints

    :param resume_path: Checkpoint path to be resumed
    """
    resume_path = str(resume_path)
    self.logger.info("Loading checkpoint: {} ...".format(resume_path))
    checkpoint = torch.load(resume_path)
    self.start_epoch = checkpoint['epoch'] + 1
    self.mnt_best = checkpoint['monitor_best']

    # load architecture params from checkpoint.
    if checkpoint['config']['arch'] != self.config['arch']:
        self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                            "checkpoint. This may yield an exception while state_dict is being loaded.")
    self.model.load_state_dict(checkpoint['state_dict'])

    # load optimizer state from checkpoint only when optimizer type is not changed.
    if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
        self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                            "Optimizer parameters not being resumed.")
    else:
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

def _save_checkpoint(self, epoch, save_best=False):
    """
    Saving checkpoints

    :param epoch: current epoch number
    :param log: logging information of the epoch
    :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
    """
    arch = type(self.model).__name__
    state = {
        'arch': arch,
        'epoch': epoch,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'monitor_best': self.mnt_best,
        'config': self.config
    }
    filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
    torch.save(state, filename)
    self.logger.info("Saving checkpoint: {} ...".format(filename))
    if save_best:
        best_path = str(self.checkpoint_dir / 'model_best.pth')
        torch.save(state, best_path)
        self.logger.info("Saving current best: model_best.pth ...")
