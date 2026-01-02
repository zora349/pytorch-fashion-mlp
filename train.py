# train.py
import torch

def train_epoch(train_load,model,loss_fn,optimizer,device):
    model.train()
    total_loss,correct,total = 0.0,0,0
    for x,y in train_load:
        x,y = x.to(device),y.to(device)
        #assert x.ndim == 4 and x.shape[1:] == (1, 28, 28), f"Bad x shape: {x.shape}"
        pred = model(x)
        loss = loss_fn(pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()* x.size(0)
        correct += (pred.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total

def test_epoch(test_load,model,loss_fn,device):
    model.eval()
    total_loss,correct,total = 0.0,0,0
    with torch.no_grad():
        for x,y in test_load:
            x,y = x.to(device),y.to(device)
            pred = model(x)
            loss = loss_fn(pred,y)

            total_loss += loss.item()* x.size(0)
            correct += (pred.argmax(1) == y).sum().item()
            total += x.size(0)

    return total_loss / total, correct / total
