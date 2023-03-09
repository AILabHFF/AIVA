# %%
import torchvision.models as models
import torch
import torch.nn as nn
#import torchvision.transforms as transforms
#import torch.optim as optim
#from torchvision import datasets
import torch.nn.functional as F
#import numpy as np
import time
import copy
import math
#import pickle

# %%
# https://github.com/revidee/pytorch-pyramid-pooling/blob/master/pyramidpooling.py
class PyramidPooling(nn.Module):
    def __init__(self, levels, mode="max"):
        """
        General Pyramid Pooling class which uses Spatial Pyramid Pooling by default and holds the static methods for both spatial and temporal pooling.
        :param levels defines the different divisions to be made in the width and (spatial) height dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :returns a tensor vector with shape [batch x 1 x n], where  n: sum(filter_amount*level*level) for each level in levels (spatial) or
                                                                    n: sum(filter_amount*level) for each level in levels (temporal)
                                            which is the concentration of multi-level pooling
        """
        super(PyramidPooling, self).__init__()
        self.levels = levels
        self.mode = mode

    def forward(self, x):
        return self.spatial_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out

    @staticmethod
    def spatial_pyramid_pool(previous_conv, levels, mode):
        """
        Static Spatial Pyramid Pooling method, which divides the input Tensor vertically and horizontally
        (last 2 dimensions) according to each level in the given levels and pools its value according to the given mode.
        :param previous_conv input tensor of the previous convolutional layer
        :param levels defines the different divisions to be made in the width and height dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :returns a tensor vector with shape [batch x 1 x n],
                                            where n: sum(filter_amount*level*level) for each level in levels
                                            which is the concentration of multi-level pooling
        """
        num_sample = previous_conv.size(0)
        previous_conv_size = [int(previous_conv.size(2)), int(previous_conv.size(3))]
        for i in range(len(levels)):
            h_kernel = int(math.ceil(previous_conv_size[0] / levels[i]))
            w_kernel = int(math.ceil(previous_conv_size[1] / levels[i]))
            w_pad1 = int(math.floor((w_kernel * levels[i] - previous_conv_size[1]) / 2))
            w_pad2 = int(math.ceil((w_kernel * levels[i] - previous_conv_size[1]) / 2))
            h_pad1 = int(math.floor((h_kernel * levels[i] - previous_conv_size[0]) / 2))
            h_pad2 = int(math.ceil((h_kernel * levels[i] - previous_conv_size[0]) / 2))
            assert w_pad1 + w_pad2 == (w_kernel * levels[i] - previous_conv_size[1]) and \
                   h_pad1 + h_pad2 == (h_kernel * levels[i] - previous_conv_size[0])

            padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2, h_pad1, h_pad2],
                                 mode='constant', value=0)
            if mode == "max":
                pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            elif mode == "avg":
                pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            else:
                raise RuntimeError("Unknown pooling type: %s, please use \"max\" or \"avg\".")
            x = pool(padded_input)
            if i == 0:
                spp = x.view(num_sample, -1)
            else:
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)

        return spp

class SPP_Net(nn.Module):
    def __init__(self, feature_extractor):
        super(SPP_Net,self).__init__()
        self.extractor = feature_extractor
        #self.fc1 = nn.Linear(in_features=2560,out_features=2) #43520
        self.fc1 = nn.Linear(in_features=5120,out_features=2) #43520

    def forward(self,x):
        x = self.extractor(x)
        #print(x.shape)
        x = PyramidPooling.spatial_pyramid_pool(x,[2,2,1,1],"max")
        #print(np.shape(x))
        output = self.fc1(x)

        return output
    
## https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856
class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average
    
    def load_to_model(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]
    
    def ema_params(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                _ = self(name, param.data)
                    
    def register_params(self, model):                
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.register(name, param.data)  

# %%
def train_model(model, dataloaders, criterion, optimizer, num_epochs=30, batch_size=100, device='cuda:0'):
    print("Start training")
    since = time.time()

    val_acc_history = []
    ema = EMA(mu=.99999)
    ema.register_params(model.cuda())
    

    best_model_wts = copy.deepcopy(model.state_dict())
    test_model = copy.deepcopy(model).to(device)
    train_model = copy.deepcopy(model).to(device)
    best_acc = 0.0
    model.to(device)
    lrs = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    
    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            idx = 0
            
            # Iterate over data.
            outputs = []
            labels = []
            model_inputs = []
            if 'phase' == 'test':
                ema.load_to_model(test_model)
                model=test_model
            elif 'phase' == 'train':
                model=train_model
            with torch.set_grad_enabled(phase == 'train'):
                for idx, (inputs, label) in enumerate(dataloaders[phase]):
                    #print(np.shape(inputs))
                    inputs = inputs.to(device)
                    #labels = torch.eye(2)[labels-1].to(device)
                    label = (label).to(device)
                    labels.append(label)
                    outputs.append(model(inputs))
                    
                    #print(labels)
                    print(f'Epoch {epoch} Phase {phase} Batch # {idx} of {len(dataloaders[phase])} running_loss {running_loss:4.4f} running_corrects {running_corrects:4.4f}', end='\r')
                    if (idx+1)%batch_size == 0:
                        
                        
                        # forward
                        # track history if only in train
                        # Get model outputs and calculate loss    
                        
                        o, l =torch.cat(outputs,dim=0), torch.stack(labels,dim=0).view(batch_size)
                        #print(o.shape, l.shape, o.argmax(1).shape)
                        loss = criterion(o,l) + ((0.5-o)**2).sum()*1e-8
                        
                        _, preds = torch.max(o, dim=1)
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()                    
                            optimizer.step()
                            # zero the parameter gradients
                            optimizer.zero_grad()
                            ema.ema_params(model)

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == l.data)
                        outputs = []
                        labels = []
                        model_inputs = []
            if 'phase' == 'test':
                test_model = model
            elif 'phase' == 'train':
                train_model = model
            print()
            epoch_loss = float(running_loss) / float(len(dataloaders[phase].dataset))
            epoch_acc = float(running_corrects) / float(len(dataloaders[phase].dataset))


            #deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'test':
                val_acc_history.append(epoch_acc)
                print('Loss: {:.4f} Acc: {:.4f} best: {:.4f}'.format(epoch_loss, epoch_acc, best_acc))
            lrs.step()


        #print()

    time_elapsed = time.time() - since
    
    #print('-' * 10)
    #print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return val_acc_history


class MeteorModel():
    def __init__(self, weithts_path, device=None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        resnet18 = models.resnet18(pretrained=True)
        feature_extractor = nn.Sequential(*(list(resnet18.children())[0:8])) #second way
        
        self.model = SPP_Net(feature_extractor).to(self.device)
        self.model.load_state_dict(torch.load(weithts_path, map_location=self.device))
        self.model.eval()
        self.model = self.model.to(self.device)

    def predict(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            output = self.model(x)
            _, preds = torch.max(output, 1)
        return preds
    
    def predict_label(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            output = self.model(x)
            _, preds = torch.max(output, 1)

        val = preds.item()
        if val == 0:
            return 'meteor'
        else:
            return 'not_meteor'
    

