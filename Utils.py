from tqdm import tqdm

def train_sinle_epoch(dataloader, model, loss_fn, optimizer):
    correct_total = 0
    size_total = 0
    with tqdm(dataloader, unit="batch") as tepoch:
        for x, y in tepoch:
            tepoch.set_description("Progress")
            # x, y = x.to(device), y.to(device)
            
            # -- Forward pass
            logits = model(x)
            
            # -- Backprop
            optimizer.zero_grad()
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            
            # -- Compute metrics
            y_pred = logits.argmax(dim=1, keepdim=True).squeeze()
            num_correct = (y_pred == y).sum().item()
            correct_total += num_correct
            size_total += len(y)
            accuracy = correct_total / size_total
            
            # -- Update the progress bar values
            tepoch.set_postfix(
                loss=loss.item(),
                acc=format(accuracy, "3.2%"),
            )