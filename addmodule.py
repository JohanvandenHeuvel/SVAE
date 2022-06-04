from torch import nn


class AddModule(nn.Module):

    def __init__(self, left_branch, right_branch):
        """
        Custom module for creating a function of the form: y = f(x) + g(x).
        Just made it for two elements but could easily be done in general list format.

        More info:
        https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html

        :param left_branch: function f
        :param right_branch: function g
        """
        super(AddModule, self).__init__()
        self.branches = nn.ModuleList([left_branch, right_branch])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        return self.branches[0](x) + self.branches[1](x)
