import argparse
import os
from model import *
from data_loader import DataLoader
from evaluator import Evaluator
from trainer import Trainer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Argument Parser for SCAN')
    parser.add_argument("--train_img_feats", dest="train_img_feats", type=str,
                        default='output_500/train_image_feats.h5')
    parser.add_argument("--dev_img_feats", dest="dev_img_feats", type=str,
                        default='output_100/dev_image_feats.h5')
    parser.add_argument("--test_img_feats", dest="test_img_feats", type=str,
                        default='output_200/test_image_feats.h5')
    parser.add_argument("--train_diff_img_feats", dest="train_diff_img_feats", type=str,
                        default='output_500/train_del_image_feats.h5')
    parser.add_argument("--dev_diff_img_feats", dest="dev_diff_img_feats", type=str,
                        default='output_100/dev_del_image_feats.h5')
    parser.add_argument("--test_diff_img_feats", dest="test_diff_img_feats", type=str,
                        default='output_200/test_del_image_feats.h5')
    parser.add_argument("--train_diff", dest="train_diff", types=str, default='output_500/CLEVR_del_diff.json')
    parser.add_argument("--test_diff", dest="test_diff", types=str, default='output_200/CLEVR_del_diff.json')
    parser.add_argument("--dev_diff", dest="dev_diff", types=str, default='output_100/CLEVR_del_diff.json')
    parser.add_argument("--diff_types", dest="diff_types", types=str, default='del')

    parser.add_argument("--hidden_dimension", dest="hidden_dimension", type=int, default=512)
    parser.add_argument("--embedding_dimension", dest="embedding_dimension", type=int, default=512)
    parser.add_argument("--output_dimension", dest="output_dimension", type=int, default=3)

    parser.add_argument("--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("--lr", dest="lr", type=float, default=0.001)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=10)
    parser.add_argument("--clip_value", dest="clip_value", type=float, default=50)
    parser.add_argument("--wdecay", dest="wdecay", type=float, default=0.00001)
    parser.add_argument("--step_size", dest="step_size", type=int, default=30)
    parser.add_argument("--gamma", dest="gamma", type=int, default=10)
    parser.add_argument("--validate_every", dest="validate_every", type=int, default=1)

    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/')
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    parser.add_argument("--mode", dest="mode", type=int, default=0)
    parser.add_argument("--model_dir", dest="model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--model_file_name", dest="model_file_name", type=str, default="best_model_weights.t7")

    parser.add_argument("--visual_feature_dimension", dest="visual_feature_dimension", type=int, default=1024)
    parser.add_argument("--regions_in_image", dest="regions_in_image", type=int, default=15 * 20)
    return parser.parse_args()


def main():
    params = parse_arguments()
    print("Constructing data loaders...")
    dl = DataLoader(params)
    evaluator = Evaluator(params, dl)
    print("Constructing data loaders...[OK]")

    if params.mode == 0:
        print("Training...")
        t = Trainer(params, dl, evaluator)
        t.train()
        print("Training...[OK]")
    elif params.mode == 1:
        print("Loading model...")
        model = DIFFSPOT(params)
        model_file_path = os.path.join(params.model_dir, params.model_file_name)
        model.load_state_dict(torch.load(model_file_path))
        if torch.cuda.is_available():
            model = model.cuda()
        print("Loading model...[OK]")

        print("Evaluating model on test set...")
        loss, acc = evaluator.get_loss_and_acc(model, is_test=True)
        print("Test Loss : {}, Test Accuracy: {}".format(loss, acc))
        print("Evaluating model on test set...[OK]")


if __name__ == '__main__':
    main()