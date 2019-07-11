from util import *
from tqdm import tqdm


class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader

    def get_loss_and_acc(self, model, is_test=False):
        """
        :param model:
        :param is_test:
        :return: loss and accuracy
        """
        model.eval()
        if is_test:
            data_loader = self.data_loader.test_data_loader
        else:
            data_loader = self.data_loader.dev_data_loader

        hits = 0
        total = 0
        losses = []
        for img, img_feat, label in tqdm(data_loader):
            model.eval()
            label = to_variable(label)
            # forward pass.
            logits = model(to_variable(img), to_variable(img_feat))
            # Compute the loss, gradients, and update the parameters by calling optimizer.step()
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, label)
            _, argmax = torch.max(logits, dim=1)
            hits += torch.sum(argmax == label).data.cpu().numpy()
            losses.append(loss.data.cpu().numpy())
            total += len(img)

        return np.asscalar(np.mean(losses)), hits / total
