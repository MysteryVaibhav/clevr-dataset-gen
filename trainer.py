import torch.utils.data
import torch.nn as nn
from model import DIFFSPOT
from timeit import default_timer as timer
from util import *
from tqdm import tqdm


def init_xavier(m):
    """
    Sets all the linear layer weights as per xavier initialization
    :param m:
    :return: Nothing
    """
    if type(m) == torch.nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.normal_(0, std)
        m.bias.data.zero_()


class Trainer:
    def __init__(self, params, data_loader, evaluator):
        self.params = params
        self.data_loader = data_loader
        self.evaluator = evaluator

    def train(self):
        model = DIFFSPOT(self.params)
        model.apply(init_xavier)
        #model.load_state_dict(torch.load('models/model_weights_5.t7'))
        loss_function = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            model = model.cuda()
            loss_function = loss_function.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.params.lr, weight_decay=self.params.wdecay)
        try:
            prev_best = 0
            for epoch in range(self.params.num_epochs):
                iters = 1
                losses = []
                start_time = timer()
                num_of_mini_batches = len(self.data_loader.training_data_loader) // self.params.batch_size
                for img, img_feat, label in tqdm(self.data_loader.training_data_loader):

                    model.train()
                    optimizer.zero_grad()
                    # forward pass.
                    logits = model(to_variable(img), to_variable(img_feat))

                    # Compute the loss, gradients, and update the parameters by calling optimizer.step()
                    loss = loss_function(logits, to_variable(label))
                    loss.backward()
                    losses.append(loss.data.cpu().numpy())
                    if self.params.clip_value > 0:
                        torch.nn.utils.clip_grad_norm(model.parameters(), self.params.clip_value)
                    optimizer.step()

                    if iters % 5 == 0:
                        tqdm.write("[{}/{}] :: Training Loss: {}".format(iters, num_of_mini_batches,
                                                                         np.asscalar(np.mean(losses))))
                    iters += 1

                if epoch + 1 % self.params.step_size == 0:
                    optim_state = optimizer.state_dict()
                    optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / self.params.gamma
                    optimizer.load_state_dict(optim_state)

                # Calculate accuracy after each epoch
                if (epoch + 1) % self.params.validate_every == 0:
                    dev_loss, dev_acc = self.evaluator.get_loss_and_acc(model, is_test=False)

                    print("Epoch {} : Training Loss: {:.5f}, Dev Loss : {}, Dev Acc : {}, Time elapsed {:.2f} mins"
                          .format(epoch + 1, np.asscalar(np.mean(losses)), dev_loss, dev_acc,
                                  (timer() - start_time) / 60))
                    if dev_acc > prev_best:
                        print("Accuracy increased....saving weights !!")
                        prev_best = dev_acc
                        torch.save(model.state_dict(), self.params.model_dir + 'best_model_weights.t7')
                else:
                    print("Epoch {} : Training Loss: {:.5f}".format(epoch + 1, np.asscalar(np.mean(losses))))
        except KeyboardInterrupt:
            print("Interrupted.. saving model !!!")
            torch.save(model.state_dict(), self.params.model_dir + '/model_weights_interrupt.t7')