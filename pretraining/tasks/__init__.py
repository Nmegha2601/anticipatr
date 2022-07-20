import torch

from datasets import build_dataset
from models import build_model

def build_task(args):
    dataset_train,dataset_test = build_dataset(args)
    model, criterion = build_model(args)


    return dataset_train, dataset_test, model, criterion

