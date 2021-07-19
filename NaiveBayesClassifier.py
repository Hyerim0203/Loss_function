#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

class NaiveBayes:
    """
    p(y|x) = p(y)*p(x|y)
    범주형 칼럼에 맞는 Naive Bayes Classifier
    """
    def fit(self,X,y):
        """
        X : 훈련 데이터의 값
        y : 훈련 데이터의 class
        훈련 데이터를 통해 p(y), p(x|y)를 구하고 저장.
        """
        n_samples, n_features = X.shape
        self._classes = np.unique(y) # class 개수
        self._n_classes = len(self._classes)

        # 각 class일 확률(p(y))
        self.yprob = [(y==i).sum()/n_samples for i in range(self._n_classes)]
        # 각 class일 때 각 feature의 각 값일 확률(p(x|y))
        self.prob = [np.zeros((n_features, int(X.max())+1)) for i in range(self._n_classes)]
        
        for c in range(self._n_classes):
            num_class = (y==c).sum()   # 해당 class 개수
            for j in range(n_features):
                subdata = X[y==c][:,j] # 해당 class를 갖는 data들의 j번째 칼럼
                for z in range(int(subdata.max())+1):
                    self.prob[c][j][z]=((subdata==z).sum())/num_class # 해당 class를 갖는 data중 j번째 칼럼이 해당 값일 확률      
        
    def predict(self, X):
        """
        X : 예측 데이터 
        예측 데이터의 값에 따라 fit 함수에서 저장해둔 각 확률을 곱해서 최종 확률을 산출
        """
        predicted = np.zeros((len(X),self._n_classes)) # 각 data의 class일 확률
        for raw in range(len(X)):
            for c in range(self._n_classes):
                prob = 1
                for j in range(X.shape[1]):
                    value = int(X[raw,j])    # 해당 data의 j번째 칼럼 값
                    prob *= self.prob[c][j][value] # class c일 때 j번째 칼럼의 값이 value일 확률
                prob *= self.yprob[c] # class c일 확률
                predicted[raw,c]=prob
        return predicted

