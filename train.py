import os
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from model import TMC
from data import Multi_view_data
import warnings
import time
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 设置 GPU

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=130)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lambda-epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0002)
    args = parser.parse_args()
    args.data_name = 'split_data.mat'
    args.data_path = r'D:\河南大学读研文件夹\毕业论文\小论文1\数据集\心肌梗死\AP'
    args.dims = [[6], [6]]
    args.views = len(args.dims)
    input_dims = [dim[0] for dim in args.dims]

    train_loader = DataLoader(Multi_view_data(args.data_path, train=True), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(Multi_view_data(args.data_path, train=False), batch_size=args.batch_size, shuffle=False)

    print(f'The number of training batches = {len(train_loader)}')

    model = TMC(classes=2, views=args.views, classifier_dims=input_dims, input_dims=input_dims, lambda_epochs=args.lambda_epochs)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def train(epoch):
        model.train()
        loss_meter = AverageMeter()
        correct_num, data_num = 0, 0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            for v_num in range(len(data)):
                data[v_num] = data[v_num].to(device)
            target = target.long().to(device)

            total_patterns = 2 ** args.views
            missing_pattern = torch.randint(1, total_patterns, (data[0].size(0),)).to(device)

            optimizer.zero_grad()
            _, loss, accuracy, preds = model(data, target, epoch, missing_pattern)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item())
            correct_num += accuracy * target.size(0)
            data_num += target.size(0)

        train_acc = correct_num / data_num
        elapsed_time = time.time() - start_time
        # print(f'Epoch {epoch} | Train Loss: {loss_meter.avg:.4f} | Train Acc: {train_acc:.4f} | Time: {elapsed_time:.2f}s')
        return train_acc, elapsed_time

    def test(epoch):
        model.eval()
        loss_meter = AverageMeter()
        correct_num, data_num = 0, 0
        all_preds, all_targets = [], []
        start_time = time.time()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                for v_num in range(len(data)):
                    data[v_num] = data[v_num].to(device)
                target = target.long().to(device)

                total_patterns = 2 ** args.views
                missing_pattern = torch.randint(1, total_patterns, (data[0].size(0),)).to(device)

                _, loss, accuracy, preds = model(data, target, epoch, missing_pattern)
                loss_meter.update(loss.item())

                correct_num += accuracy * target.size(0)
                data_num += target.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        test_acc = correct_num / data_num
        elapsed_time = time.time() - start_time
        # print(f'====> Epoch {epoch} | Test Loss: {loss_meter.avg:.4f} | Test Acc: {test_acc:.4f} | Time: {elapsed_time:.2f}s')
        return loss_meter.avg, test_acc, elapsed_time, all_preds, all_targets

    best_acc = 0.0
    best_preds, best_targets = None, None
    start_total_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_acc, train_time = train(epoch)

        if epoch % 5 == 0:
            test_loss, test_acc, test_time, all_preds, all_targets = test(epoch)
            scheduler.step(test_loss)
            if test_acc > best_acc:
                best_acc = test_acc
                best_preds = all_preds
                best_targets = all_targets
                torch.save(model.state_dict(), f'best_model.pth')
                print(f'====> New Best Model Saved with ACC: {best_acc:.4f}')

    end_total_time = time.time()
    total_elapsed_time = end_total_time - start_total_time
    print(f'Total Training and Testing Time: {total_elapsed_time:.2f}s')
    print(f'Best Test ACC: {best_acc:.4f}')

    # 绘制混淆矩阵
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    best_preds = np.array(best_preds)
    best_targets = np.array(best_targets)
    cm = confusion_matrix(best_targets, best_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Best Model)")
    plt.savefig("confusion_matrix.png")
    plt.show()