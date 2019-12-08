import numpy as np 

class DataSetBacteria (object): 
    def __init__(self):
        self.x=None 
        self.y=None 
        self.classes=None
        self.w=330
        self.h=330
        self.c=3

    def load (self,path,v=False):
        files=np.load(path)
        self.x=files['imgs']
        self.y=files['labels']
        self.classes=files['classes'].tolist()
        self.x = self.x.reshape(self.x.shape[0],self.h,self.w,self.c)
        self.label2Int()
        self.shuffle()
        
        if v:
            print(self.x.shape)
            print(self.y.shape)
            print(self.classes)

    def getDataSet(self):
        return (self.x,self.y)

    def getClasses(self):
        return self.classes
    
    def label2Int(self):
        for i in range(0,len(self.y)):
            self.y[i] = self.classes.index(self.y[i])


    def shuffle(self):
        random_index = np.arange(self.x.shape[0])
        np.random.shuffle(random_index)
        self.x = self.x[random_index]
        self.y = self.y[random_index]

if __name__=='__main__':
    train = DataSetBacteria()
    train.load('./bin/test_qtd_classes(3).npz',True)
    
