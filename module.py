import torch
from torch import nn
from torchvision.models.resnet import resnet18
import pytorch_lightning as pl


class RaceClassifier(pl.LightningModule):
    def __init__(self, num_classes, net_type='resnet', pretrained=False, freeze_layers=False, data_module=None, **kwargs):
        super().__init__()
        self.kwargs = kwargs

        if net_type == 'resnet':
            if not pretrained:
                self.model = resnet18(num_classes=num_classes)
            else:
                self.model = resnet18(pretrained=True)
                self.model.fc = nn.Linear(512, num_classes)

                if freeze_layers:
                    for name, param in self.model.named_parameters():
                        if not name.startswith('layer4'):
                            param.requires_grad = False
                        else:
                            break
                            
        if data_module is not None:
            self.n_samples_in_train = len(data_module.dset_train)    # __len__() must be implemented correctly
        else:
            self.n_samples_in_train = float('inf')
        
        self.train_average_stats = None
        self.epoch_no = 0
        self.log_images_global_step = 0
        self.epoch_real = 0
        self.total_n_samples_seen = 0
        
    def forward(self, x):
        logits = self.model(x)
        pred = torch.softmax(logits, dim=1)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
