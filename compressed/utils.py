import os
import shutil
import torch


def ensure_dir(path, erase=False):
    if os.path.exists(path) and erase:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)


def test(model, dataloader):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    model = model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = torch.FloatTensor(X).cuda(), torch.LongTensor(y).cuda()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print("Acc: {0}, Loss: {1}".format(correct, test_loss))


def eval_model(model, dataloader, n_sample=None):
    correct1, correct5 = 0, 0
    n_passed = 0
    model = model.eval()

    n_sample = len(dataloader) if n_sample is None else n_sample
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataloader):
            n_passed += len(data)
            data = data.cuda()
            index_target = target.cuda()
            output = model(data)
            bs = output.size(0)
            idx_pred = output.data.sort(1, descending=True)[1]

            idx_gt1 = index_target.expand(1, bs).transpose_(0, 1)
            idx_gt5 = idx_gt1.expand(bs, 5)

            correct1 += idx_pred[:, :1].eq(idx_gt1).sum()
            correct5 += idx_pred[:, :5].eq(idx_gt5).sum()

            if idx >= n_sample - 1:
                break

    acc1 = correct1 * 1.0 / n_passed
    acc5 = correct5 * 1.0 / n_passed
    return acc1, acc5