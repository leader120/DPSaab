import numpy as np
import Fashion_Mnist_FF.cwsaab as cwsaab

class Pixelhop2(cwsaab):
    def __init__(self, depth=1, TH1=0.01, TH2=0.001, SaabArgs=None, shrinkArgs=None, concatArg=None):
        super().__init__(depth=depth, energyTH=TH1, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg={'func':lambda X, concatArg: X})
        self.TH1 = TH1
        self.TH2 = TH2
        self.concatArg = concatArg

    def select_(self, X):
        for i in range(self.depth):
            X[i] = X[i][:, :, :, self.Energy[i] >= self.TH2]
        return X

    def fit(self, X):
        X, _ = super().fit(X)
        X = self.select_(X)
        return self.concatArg['func'](X, self.concatArg)

    def transform(self, X):
        X, _ = super().transform(X)
        X = self.select_(X)
        return self.concatArg['func'](X, self.concatArg)

if __name__ == "__main__":
    # example useage
    from sklearn import datasets
    from skimage.util import view_as_windows

    # example callback function for collecting patches and its inverse
    def Shrink(X, shrinkArg):
        win = shrinkArg['win']
        X = view_as_windows(X, (1,win,win,1), (1,win,win,1))
        return X.reshape(X.shape[0], X.shape[1], X.shape[2], -1)

    # example callback function for how to concate features from different hops
    def Concat(X, concatArg):
        return X

    # read data
    import cv2
    print(" > This is a test example: ")
    digits = datasets.load_digits()
    X = digits.images.reshape((len(digits.images), 8, 8, 1))
    print(" input feature shape: %s"%str(X.shape))

    # set args
    SaabArgs = [{'num_AC_kernels':-1, 'needBias':False, 'useDC':True, 'batch':None},
                {'num_AC_kernels':-1, 'needBias':True, 'useDC':True, 'batch':None}]
    shrinkArgs = [{'func':Shrink, 'win':2},
                {'func': Shrink, 'win':2}]
    concatArg = {'func':Concat}

    print(" --> test inv")
    print(" -----> depth=1")
    p2 = Pixelhop2(depth=1, TH1=0.01, TH2=0.001, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)
    output = p2.fit(X)
    output = p2.transform(X)
    print(" -----> depth=2")
    p2 = Pixelhop2(depth=2, TH1=0.01, TH2=0.001, SaabArgs=SaabArgs, shrinkArgs=shrinkArgs, concatArg=concatArg)
    output = p2.fit(X)
    output = p2.transform(X)
    print(output[0].shape, output[1].shape)
    print("------- DONE -------\n")