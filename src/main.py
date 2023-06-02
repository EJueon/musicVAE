import argparse
from omegaconf import OmegaConf

import os 
import sys
import torch 

from utils.process_utils import preprocess, get_num_classes
from models.dataloader import load_dataloader
from models.musicVAE import MusicVAE
from models.metrics import ELBO_loss, get_accuracy
from utils.trainer import Trainer
from utils import save_dataset, load_dataset, setdir

def preprocess_data(conf, dir_path, extensions):
    print("preprocess dataset...")
    dataset = preprocess(dir_path, extensions)
    setdir(conf.save_dir)
    print(f"save dataset..train:dev = {1-conf.dev_ratio}:{conf.dev_ratio}")
    train_path, dev_path = save_dataset(dataset, save_path=conf.save_dir, dev_ratio=conf.dev_ratio)
    print(f"save train dataset to {os.path.abspath(train_path)}")
    print(f"save dev dataset to {os.path.abspath(dev_path)}")
    
def train(conf):
    num_classes = get_num_classes()
    train_dataloader = load_dataloader(conf, conf.train_dataset_path, num_classes)
    dev_dataloader = load_dataloader(conf, conf.dev_dataset_path, num_classes)
    model = MusicVAE(input_size=2**num_classes, 
                     enc_latent_dim=conf.enc_latent_dim,
                     conductor_dim=conf.conductor_dim,
                     enc_hidden_size=conf.enc_hidden_size,
                     dec_hidden_size=conf.dec_hidden_size,
                     conductor_hidden_size=conf.conductor_hidden_size,
                     enc_num_layers=conf.enc_num_layers,
                     dec_num_layers=conf.dec_num_layers,
                     conductor_num_layers=conf.conductor_num_layers)

    trainer = Trainer(conf, model, criterion=ELBO_loss, eval_func=get_accuracy)
    trainer.train(train_dataloader, dev_dataloader)

def generate(conf, model_path, midi_path):
    model = MusicVAE(input_size=2**num_classes, 
                     enc_latent_dim=conf.enc_latent_dim,
                     conductor_dim=conf.conductor_dim,
                     enc_hidden_size=conf.enc_hidden_size,
                     dec_hidden_size=conf.dec_hidden_size,
                     conductor_hidden_size=conf.conductor_hidden_size,
                     enc_num_layers=conf.enc_num_layers,
                     dec_num_layers=conf.dec_num_layers,
                     conductor_num_layers=conf.conductor_num_layers)
    
    model.load_state_dict(torch.load(model_path)).to('cuda')
    
    # preprocess_data = preprocess_one(midi_path)
    # pred, _, _ = model(preprocess_data)
    # pred = torch.argmax(pred, dim=2)
    # print(pred)
    # TODO : numpy array to MIDI 
    

def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--option",
        required=True,
        choices=["preprocess", "train", "generate"],
        help="set preprocess, train mode",
    )
    arg_parser.add_argument(
        "--config_path",
        required=False,
        default="../configs/basic_config.yaml",
        help="configuration path",
    )
    arg_parser.add_argument(
        "--dataset_path",
        required=False,
        default="../dataset",
        help="dataset directory path",
    )
    arg_parser.add_argument(
        "--model_path",
        required=False,
        default="../data/ckpt/epoch_50.pt",
        help="model checkpoint path",
    )
    arg_parser.add_argument(
        "--midi_path",
        required=False,
        default="../data/ckpt/epoch_50.pt",
        help="midi file path",
    )
    return arg_parser.parse_args()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Our program supports only CUDA enabled machines")
        sys.exit(1)
    args = get_args()

    conf = OmegaConf.load(args.config_path)
    if args.option == "preprocess":
        preprocess_data(conf, args.dataset_path, conf.extensions)
    elif args.option == "train":
        train(conf)
    elif args.option == "generate":
        generate(conf, args.model_path)
        