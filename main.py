import os
import argparse
import data_loader
import models.CNN
import models.lenet

def main(args):
    # load data
    src_dataset = args.source_data[0]
    tgt_dataset = args.target_data[0]
    print(f'source dataset: {src_dataset}')
    print(f'target dataset: {tgt_dataset}')
    if src_dataset == 'mnist':
        src_loader = data_loader.mnist_loader_train
    
    if tgt_dataset == 'mnist':
        tgt_loader = data_loader.mnist_loader_train

    model = args.model[0]
    print(f'model: {model}')
    if model == 'cnn':
        model_train = models.CNN.CNNClassifier()
    if model == 'lenet':
        model_train = models.lenet.LeNet5()

    print(model_train)

    # train data
    import torch.optim as optim
    import torch.nn as nn

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_train.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(src_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model_train(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999: # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i+1, running_loss / 2000))
                running_loss = 0.0
            
    print('Finished Training')    

    # test data
    model_train.eval()
    test_loss=0
    correct=0
    for data, target in tgt_loader:
        #data, target = Variabledata, volatile=True), Variable(target)
        output = model_train(data)

        test_loss += criterion(output, target).data

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(tgt_loader.dataset)
    print('\nTest est: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'.
    format(test_loss, correct, len(tgt_loader.dataset), 
    100.*correct/len(tgt_loader.dataset)))

    dataiter = iter(tgt_loader)
    images, labels = dataiter.next()

    # print images
    
    


    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This program is for experimenting domain adaptation methods.')

    parser.add_argument('--source-data', 
                        type=str, 
                        choices=['mnist', 'mnist-i', 'mnist-m', 'svhn'],
                        required=True,
                        nargs=1,
                        help='Source data')
    parser.add_argument('--target-data', 
                        type=str, 
                        choices=['mnist', 'mnist-i', 'mnist-m', 'svhn'],
                        required=True,
                        nargs=1,
                        help='Target data')
    parser.add_argument('--model', 
                        type=str, 
                        required=True,
                        choices=['cnn', 'dann', 'lenet', ],
                        nargs=1,
                        help='model')
    
    args = parser.parse_args()
    
    main(args)