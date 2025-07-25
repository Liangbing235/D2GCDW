import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
class Multi_view_data(Dataset):
    """
    load multi-view data
    """

    def __init__(self, root, train=True):
        """
        :param root: data name and path
        :param train: load training set or test set
        """
        super(Multi_view_data, self).__init__()
        # self.root = r'D:\河南大学读研文件夹\毕业论文\小论文1\模型代码\TMC-main\TMC ICLR\datasets\modified_Caltech101-20'
        self.root = r'D:\河南大学读研文件夹\毕业论文\小论文1\数据集\心肌梗死\AP\split_data'
        self.train = train
        data_path = self.root + '.mat'

        dataset = sio.loadmat(data_path)
        view_number = int((len(dataset) - 5) / 2) #计算 dataset 的长度减去 5 后的一半，并将结果取整为整数。
        self.X = dict()#初始化一个空字典，并将其赋值给对象的属性 self.X。
        if train:#根据 train 参数决定加载训练集或测试集的数据：
            for v_num in range(view_number):
                self.X[v_num] = normalize(dataset['x' + str(v_num + 1) + '_train'])
            y = dataset['gt_train']
        else:
            for v_num in range(view_number):
                self.X[v_num] = normalize(dataset['x' + str(v_num + 1) + '_test'])
            y = dataset['gt_test']

        if np.min(y) == 1:
            y = y - 1
        tmp = np.zeros(y.shape[0])
        y = np.reshape(y, np.shape(tmp))
        self.y = y#这段代码的作用是对标签数组 y 进行预处理，确保其最小值为 0，并将其转换为与 tmp 相同形状的一维数组，最终赋值给 self.y。
        self.view_number = view_number  # 新增：记录视图数

    def __getitem__(self, index):#这段代码定义了 __getitem__ 方法，用于根据索引 index 获取数据样本和对应的目标值：
        data = dict()#遍历所有视图（self.X 的键），提取对应视图的第 index 个数据样本，转换为 float32 类型，并存入 data。
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
        target = self.y[index]
        return data, target

    def __len__(self):
        return len(self.X[0])


def normalize(x, min=0):
    if min == 0:
        scaler = MinMaxScaler((0, 1))
    else:  # min=-1
        scaler = MinMaxScaler((-1, 1))
    norm_x = scaler.fit_transform(x)
    return norm_x
