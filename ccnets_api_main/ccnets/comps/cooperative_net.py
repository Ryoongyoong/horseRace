'''
    CCNet's configurators implementations.

    Author:
        PARK, JunHo, junho.zerotoone@gmail.com
'''
import torch
from .explainer import Explainer
from .reasoner import Reasoner
from .producer import Producer
from ..utils.setting import set_train_param_args

class CooperativeNet():
    def __init__(self, args, explainer_net, reasoner_net, producer_net):
        set_train_param_args(self, args)
        self.model_names = ["explainer", "reasoner", "producer"]
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
        self.ngpu = args.ngpu
        self.explainer =  Explainer.create(args, explainer_net).to(self.device)
        self.reasoner =  Reasoner.create(args, reasoner_net).to(self.device)
        self.producer =  Producer.create(args, producer_net).to(self.device)
        self._models = [self.explainer, self.reasoner, self.producer]

    def get_models(self):
        return self._models

    def explain(self, x):
        with torch.no_grad():
            e = self.explainer(x)
        return e

    def reason(self, x, e):
        with torch.no_grad():
            y = self.reasoner(x, e)
        return y

    def produce(self, y, e):
        with torch.no_grad():
            x = self.producer(y, e)
        return x

    def infer(self, x):
        with torch.no_grad():
            e = self.explainer(x)
            y = self.reasoner(x, e)
        return y

    def generate(self, y, noise):
        with torch.no_grad():
            x_generated = self.producer(y, noise)
        return x_generated

    def reconstruct(self, x):
        with torch.no_grad():
            e = self.explainer(x)
            y_inferred = self.reasoner(x, e)
            x_reconstructed = self.producer(y_inferred, e)
        return x_reconstructed

