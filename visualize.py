from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation 
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch import nn


class Visualize:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')

        base_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(root=args.data_root,
                                         train=True,
                                         download=True,
                                         transform=base_transforms)

        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           drop_last=True)

        val_dataset = datasets.CIFAR10(root=args.data_root,
                                       train=False,
                                       download=True,
                                       transform=base_transforms)

        self.val_dataloader = DataLoader(val_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.num_workers,
                                         pin_memory=True,
                                         drop_last=True)

    def visualize(self):
        """Extract features from validation split and search on train split features."""
        n_data = self.train_dataloader.dataset.data.shape[0]
        feat_dim = self.args.feat_dim

        self.model.eval()
        if str(self.device) == 'cuda':
            torch.cuda.empty_cache()

        train_features = torch.zeros([n_data, feat_dim])
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.train_dataloader):
                inputs = inputs.to(self.device)
                batch_size = inputs.size(0)

                # forward
                features = self.model(inputs)
                features = nn.functional.normalize(features)
                train_features[batch_idx * batch_size:batch_idx * batch_size + batch_size, :] = features.data.cpu()

            train_labels = torch.LongTensor(self.train_dataloader.dataset.targets)
            
        pca = PCA(n_components = 3)
        train_features = pca.fit_transform(train_features)
        print(pca.explained_variance_ratio_)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        for i in range(10):
            point = train_features[train_labels == i][:self.args.visualize_num]
            ax.scatter(point[:, 0], point[:, 1], point[:, 2], label=i)

        plt.legend()
        # plt.show() 

        def animate(i):
            ax.view_init(elev=30., azim=i)
            ax.view_init(elev=30., azim=i)
            return fig,

        # Animate
        anim = animation.FuncAnimation(fig, animate, frames=360, interval=20, blit=True)
        # Save
        anim.save('mpl3d_scatter.gif', fps=30) 


def test():
    # 데이터셋 로드
    iris = load_iris()
    df = pd.DataFrame(data= np.c_[iris.data])

    # 데이터셋 정규화
    scaler = StandardScaler()    
    scaler.fit(df)
    df_scaled = scaler.transform(df)
    print(df_scaled.shape)

    # 데이터 프레임으로 자료형 변환 및 target class 정보 추가
    df_scaled = pd.DataFrame(df_scaled, columns= ['sepal length', 'sepal width', 'petal length', 'petal width'])
    df_scaled['target'] = iris.target

    print(df_scaled)

    # 2차원으로 차원 축소, target 정보는 제외
    pca = PCA(n_components = 2)
    pca.fit(df_scaled.iloc[:,:-1])
    
    # pca transform 후 데이터프레임으로 자료형 변경
    df_pca = pca.transform(df_scaled.iloc[:,:-1])
    df_pca = pd.DataFrame(df_pca, columns = ['component 0', 'component 1'])

    print(df_pca)
    print(pca.explained_variance_ratio_)

    # class target 정보 불러오기 
    df_pca['target'] = df_scaled['target']

    # target 별 분리
    df_pca_0 = df_pca[df_pca['target'] == 0]
    df_pca_1 = df_pca[df_pca['target'] == 1]
    df_pca_2 = df_pca[df_pca['target'] == 2]

    # target 별 시각화
    plt.scatter(df_pca_0['component 0'], df_pca_0['component 1'], color = 'orange', alpha = 0.7, label = 'setosa')
    plt.scatter(df_pca_1['component 0'], df_pca_1['component 1'], color = 'red', alpha = 0.7, label = 'versicolor')
    plt.scatter(df_pca_2['component 0'], df_pca_2['component 1'], color = 'green', alpha = 0.7, label = 'virginica')

    plt.xlabel('component 0')
    plt.ylabel('component 1')
    plt.legend()
    plt.show()