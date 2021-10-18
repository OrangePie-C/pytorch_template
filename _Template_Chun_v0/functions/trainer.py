import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter


def _train_epoch(self, epoch):
    """
    Training logic for an epoch

    :param epoch: Integer, current training epoch.
    :return: A log that contains average loss and metric in this epoch.
    """
    self.model.train()
    self.train_metrics.reset()
    for batch_idx, (data, target) in enumerate(self.data_loader):
        data, target = data.to(self.device), target.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()

        self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
        self.train_metrics.update('loss', loss.item())
        for met in self.metric_ftns:
            self.train_metrics.update(met.__name__, met(output, target))

        if batch_idx % self.log_step == 0:
            self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                epoch,
                self._progress(batch_idx),
                loss.item()))
            self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        if batch_idx == self.len_epoch:
            break
    log = self.train_metrics.result()

    if self.do_validation:
        val_log = self._valid_epoch(epoch)
        log.update(**{'val_' + k: v for k, v in val_log.items()})

    if self.lr_scheduler is not None:
        self.lr_scheduler.step()
    return log

def _valid_epoch(self, epoch):
    """
    Validate after training an epoch

    :param epoch: Integer, current training epoch.
    :return: A log that contains information about validation
    """
    self.model.eval()
    self.valid_metrics.reset()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(self.valid_data_loader):
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)
            loss = self.criterion(output, target)

            self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
            self.valid_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(output, target))
            self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

    # add histogram of model parameters to the tensorboard
    for name, p in self.model.named_parameters():
        self.writer.add_histogram(name, p, bins='auto')
    return self.valid_metrics.result()

def train(train_data_loader, valid_data_loader, model, criterion, metrics, optimizer, lr_scheduler, config=None):

    if resume:
        # proceed resume



    """
    Full training logic
    """
    not_improved_count = 0
    for epoch in range(self.start_epoch, self.epochs + 1):
        result = self._train_epoch(epoch)

        # save logged informations into log dict
        log = {'epoch': epoch}
        log.update(result)

        # print logged informations to the screen
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

        # evaluate model performance according to configured metric, save best checkpoint as model_best
        best = False
        if self.mnt_mode != 'off':
            try:
                # check whether model performance improved or not, according to specified metric(mnt_metric)
                improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                           (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
            except KeyError:
                self.logger.warning("Warning: Metric '{}' is not found. "
                                    "Model performance monitoring is disabled.".format(self.mnt_metric))
                self.mnt_mode = 'off'
                improved = False

            if improved:
                self.mnt_best = log[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count > self.early_stop:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                 "Training stops.".format(self.early_stop))
                break

        if epoch % self.save_period == 0:
            self._save_checkpoint(epoch, save_best=best)

