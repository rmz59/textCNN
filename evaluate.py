def evaluate(model, dev_data_iter max_sent_length, device):
    was_training = model.training
    model.eval()
    val_loss = 0
    with torch.no_grad():   
        step = 0
        for batch in iter(train_iter):
            feature, target = batch.text.T, batch.label.squeeze(0)
            step += 1
            res = model(feature) 
            loss = cross_entropy(res, target)
            val_loss += loss / step
    if was_training:
        model.train()
    return val_loss



