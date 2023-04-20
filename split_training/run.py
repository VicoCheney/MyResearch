from argparse import ArgumentParser
import os
from metrics import evaluate
from split_training.traing_loop import train


def main_parser(parser):

    parser.add_argument("--save_dir", type=str, default=os.path.join(os.getcwd(), 'save_dir'), help="saving models")
    parser.add_argument("--tmp_dir", type=str, default=os.path.join(os.getcwd(), 'tmp_dir'), help="saving ddp tmp files")
    parser.add_argument("--log_dir", type=str, default=os.path.join(os.getcwd(), 'log_dir'), help="saving logs")
    parser.add_argument("--bert_epochs", type=int, default=5, help="num epoch")
    parser.add_argument("--bart_epochs", type=int, default=20, help="num epoch")
    parser.add_argument('--compresser_model', type=str, default='roberta-base', help='name of pretrained models')
    parser.add_argument('--summarizer_model', type=str, default='facebook/bart-large-cnn', help='name of pretrained models')
    parser.add_argument('--version', type=int, default=0, help='the version to save or restore')
    parser.add_argument('--num_samples', type=str, default='10,10', help='num of continous, discrete random samples and promising samples')
    parser.add_argument('--times', type=str, default='4,8', help='memreplay times')
    parser.add_argument('--batch_size_inference', type=int, default=64, help='batch_size in memreplay')
    parser.add_argument("--gpus", type=int, default=0, help="available gpus")
    parser.add_argument('--train_source', type=str, default=os.path.join(os.getcwd(), '../data', 'arxiv_mini_train.pkl'), help='training dataset')
    parser.add_argument('--test_source', type=str, default=os.path.join(os.getcwd(), '../data', 'arxiv_mini_test.pkl'), help='test dataset')
    parser.add_argument('--validation_source', type=str, default=os.path.join(os.getcwd(), '../data', 'arxiv_mini_validation.pkl'), help='validation dataset')
    parser.add_argument('--only_train', action='store_true')
    parser.add_argument('--only_evaluate', action='store_true')
    parser.add_argument('--lr1', type=float, default=4e-5, help='learning rate of BERT')
    parser.add_argument('--bert_batch_size', type=int, default=32, help='gradient batch_size')
    parser.add_argument('--lr2', type=float, default=4e-5, help='learning rate of BART')
    parser.add_argument('--bart_batch_size', type=int, default=2, help='gradient batch_size')

    return parser


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser = main_parser(parser)
    config = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not config.only_evaluate:
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡Training Started!⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡")
        train(config)
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡Training Finished!⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡")

    if not config.only_train:
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡Evaluation Started!⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡")
        evaluate(config, mode='test')
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡Evaluation Finished!⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡")