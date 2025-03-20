import torch
import torch.nn as nn
import torch.optim as optim # 优化器
from torch.utils.data import DataLoader # 数据加载
from torchvision import datasets,transforms # 数据集和数据变换
from tqdm import tqdm # 训练进度条
import os
from model.cnn import simplecnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 设备选择cpu or gpu

# 对数据集做变换
train_transformer = transforms.Compose([
    transforms.Resize([224, 224]), # 将数据裁剪为224*224大小
    transforms.ToTensor(), # 把图片转换为 tensor张量 0-1像素值
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 标准化(均值，方差) (0-0.5)/0.5到(1-0.5)/0.5（0-1 -> -1-1)
])

test_transformer = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 定义数据集加载类
trainset = datasets.ImageFolder(root=os.path.join(r"deeplearning-model\dataset\COVID_19_Radiography_Dataset", "train"),
                                 transform=train_transformer) # 训练集做图形变换

testset = datasets.ImageFolder(root=os.path.join(r"deeplearning-model\dataset\COVID_19_Radiography_Dataset", "test"),
                                transform=test_transformer) # 测试集做图形变换

# 定义数据加载器
# 训练集加载器 trainset传入训练集，batch批次训练图像数量，num_workers数据加载多线程，0表示不打开，shuffle是否打乱数据
train_loader = DataLoader(trainset, batch_size=32, num_workers=0, shuffle=True) 

# 测试集加载器
test_loader = DataLoader(testset, batch_size=32, num_workers=0, shuffle=False)

def train(model, train_loader, criterion, optimizer, num_epochs):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"eopch:{epoch+1}/{num_epochs}", unit="batch"): # 训练时可看到对应的epoch和batch
            inputs, labels = inputs.to(device), labels.to(device) # 数据和标签传入设备
            optimizer.zero_grad() # 梯度清零
            ouputs = model(inputs) # 前向传播
            loss = criterion(ouputs, labels) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            running_loss += loss.item() * inputs.size(0) # 用loss乘inputs.size(0)得到一个batch(inputs.size(0))的loss
        epoch_loss = running_loss / len(train_loader.dataset) # 总损失除以数据集大小得到每轮的损失
        print(f"epoch[{epoch+1}/{num_epochs}, Train_loss{epoch_loss:.4f}]")

        accuracy = evaluate(model, test_loader, criterion)
        if accuracy > best_acc:
            best_acc = accuracy
            save_model(model, save_path)
            print("model saved with best acc", best_acc)

    
def evaluate(model, test_loader, criterion):
    model.eval() # 指定模型为评估模式
    test_loss = 0.0 # 初始化测试损失
    correct = 0 # 初始化正确样本数量为0
    total = 0 # 初始化总样本数量为0
    with torch.no_grad(): # 不计算梯度
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1) # 返回每行最大值和索引
            total += labels.size(0) # 计算总样本数量
            correct += (predicted == labels).sum().item() # 计算正确样本数量
    
    avg_loss = test_loss / len(test_loader.dataset) # 计算平均损失
    accuracy = 100.0 * correct / total # 计算准确率
    print(f"Test_loss:{avg_loss:.4f}, Accuracy:{accuracy:.2f}%")
    return accuracy

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    num_epochs = 10
    learning_rate = 0.001
    num_class = 4
    save_path = r"deeplearning-model\model_pth\best.pth"
    model = simplecnn(num_class).to(device) # 实例化模型
    criterion = nn.CrossEntropyLoss() # 交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # 优化器
    train(model, train_loader, criterion, optimizer, num_epochs) # 训练模型
    evaluate(model, test_loader, criterion) # 测试模型