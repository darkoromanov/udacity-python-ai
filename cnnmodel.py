import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import optim
from torch.autograd import Variable


class PreTrainedCNN:

    def __init__(self, arch, gpu, hidden_units, pretrained):
        self.model = getattr(models, arch)(pretrained=pretrained)
        self.gpu = gpu
        self.arch = arch
        self.hidden_units = hidden_units
        for param in self.model.parameters():
            param.requires_grad = False
        
        classifier = nn.Sequential(OrderedDict([ 
                                    ('fc1', nn.Linear(25088, hidden_units)), 
                                    ('relu', nn.ReLU()), 
                                    ('fc2', nn.Linear(hidden_units, 500)), 
                                    ('relu', nn.ReLU()), 
                                    ('fc3', nn.Linear(500, 102)), 
                                    ('output', nn.LogSoftmax(dim=1))
                                ]))
            
        self.model.classifier = classifier
        self.criterion = nn.NLLLoss()

        self.loading_shapes = ['|', '/', '-', '\\']
        self.loading_idx = 0
        self.loading_shapes_len = 4 #save few nanoseconds
        
        if gpu:
            self.model.cuda()

    def save(self, save_dir, class_to_idx):
        to_serialize = {
            "arch": self.arch,
            "hidden_units": self.hidden_units,
            "state_dict": self.model.state_dict(),
            "class_to_idx": class_to_idx
        }
        torch.save(to_serialize, save_dir + '/checkpoint.pth')
        
    def train(self, loader, lr, epochs):
        steps = 0
        running_loss = 0
        print_every = 100
        
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr)

        print("Training started")
        for e in range(epochs):
            self.model.train()
            for ii, (inputs, labels) in enumerate(loader):   

                self.loading()
                steps += 1                
                
                self.optimizer.zero_grad()
                inputs, labels = Variable(inputs), Variable(labels)

                if self.gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                outputs = self.model.forward(inputs)

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                
                if steps % print_every == 0:
                    print("Epoch: {}/{}... ".format(e+1, epochs),
                        "Loss: {:.4f}".format(running_loss/print_every))
                    
                    running_loss = 0
                                       
        print ("Training complete")

    def validate(self, loader):
        self.model.eval()

        accuracy = 0
        test_loss = 0
        steps = 0
    
        print("Validating model")
        
        for ii, (inputs, labels) in enumerate(loader):
            self.loading()
            steps += 1

            self.optimizer.zero_grad()
            inputs, labels = Variable(inputs), Variable(labels)
            
            if self.gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            output = self.model.forward(inputs)
            test_loss += self.criterion(output, labels).data.item()
            ps = torch.exp(output).data

            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()

        print("Test Loss: {:.3f}.. ".format( test_loss / len(loader) ),
              "Test Accuracy: {:.3f}".format( accuracy / len(loader)) )

    def loading(self, index=None, tot=None):
        perc = ""
        if index != None and tot != None:
            n = round(index / tot * 100)
            perc = '   {0:02d}%'.format(n)
            
        print("\r" + self.loading_shapes[self.loading_idx % self.loading_shapes_len] + perc, end = '\r')
        self.loading_idx += 1
