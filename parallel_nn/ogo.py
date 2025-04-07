from typing import Callable, List, Dict, Type, Any
from pydantic import BaseModel, ConfigDict
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim.optimizer import Optimizer
import torch
from tqdm import tqdm


class ExperimentBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: nn.Module
    device: str
    loss: nn.Module
    dataloader_test: DataLoader
    history: List[Dict[str, float]]
    model_test: Callable


class GeneralExperimentModel(ExperimentBaseModel): ...


class TrainableExperimentModel(ExperimentBaseModel):
    optimizer: Optimizer
    dataloader_train: DataLoader


class CopyExperimentModel(TrainableExperimentModel):
    submodules: List[str]
    epochs: int


class ReferenceExperimentModel(TrainableExperimentModel):
    optimizer: Optimizer
    dataloader_train: DataLoader


class Experiment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_reference: ReferenceExperimentModel
    model_general: GeneralExperimentModel
    model_copies: List[CopyExperimentModel]
    model_train_step: Callable[[TrainableExperimentModel],None]
    epochs: int
    model_test:Callable


class ExperimentDescription(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Type[nn.Module]
    # loss: Type[nn.Module]
    loss: Any
    
    optimizer: Type[Optimizer]
    optimizer_params: Dict[str, Any]
    dataset_train: Dataset
    dataset_test: Dataset
    batch_size: int
    epochs: int
    model_copies_epochs: int = 1
    model_copies_submodules: List[List[str]]
    model_reference_device: str
    model_general_device: str
    model_copies_devices: List[str]
    model_train_step: Callable[[TrainableExperimentModel],None]
    model_test: Callable



def generate_experiment(description: ExperimentDescription) -> Experiment:
    # reference
    reference_model = description.model().to(description.model_reference_device)
    reference = ReferenceExperimentModel(
        model=reference_model,
        device=description.model_reference_device,
        loss=description.loss(),
        dataloader_test=DataLoader(
            description.dataset_test, batch_size=description.batch_size, shuffle=False
        ),
        history=list(),
        optimizer=description.optimizer(
            reference_model.parameters(), **description.optimizer_params
        ),
        dataloader_train=DataLoader(
            description.dataset_train, batch_size=description.batch_size, shuffle=True
        ),
        model_test=description.model_test
    )

    # general
    general_model = description.model()
    general_model.load_state_dict(reference_model.state_dict())
    general_model.to(description.model_general_device)
    general = GeneralExperimentModel(
        model=general_model,
        device=description.model_general_device,
        loss=description.loss(),
        dataloader_test=DataLoader(
            description.dataset_test, batch_size=description.batch_size, shuffle=False
        ),
        history=list(),
        model_test=description.model_test
    )

    # copies
    copies = list()
    for i in range(len(description.model_copies_submodules)):
        copy_model = description.model()
        copy_model.load_state_dict(reference_model.state_dict())
        copy_model.to(description.model_copies_devices[i])
        copy = CopyExperimentModel(
            model=copy_model,
            device=description.model_copies_devices[i],
            loss=description.loss(),
            dataloader_test=DataLoader(
                description.dataset_test,
                batch_size=description.batch_size,
                shuffle=False,
            ),
            history=list(),
            optimizer=description.optimizer(
                copy_model.parameters(), **description.optimizer_params
            ),
            dataloader_train=DataLoader(
                description.dataset_train,
                batch_size=description.batch_size,
                shuffle=True,
            ),
            submodules=description.model_copies_submodules[i],
            epochs=description.model_copies_epochs,
            model_test=description.model_test
        )
        copies.append(copy)

    return Experiment(
        model_reference=reference,
        model_general=general,
        model_copies=copies,
        epochs=description.epochs,
        model_train_step=description.model_train_step,
        model_test=description.model_test
    )


def model_train_step(m: TrainableExperimentModel):
    m.model.train()
    for batch, (X, y) in enumerate(m.dataloader_train):
        X, y = X.to(m.device), y.to(m.device)

        pred = m.model(X)
        loss = m.loss(pred, y)

        loss.backward()
        m.optimizer.step()
        m.optimizer.zero_grad()


def model_test(m: GeneralExperimentModel):
    size = len(m.dataloader_test.dataset)
    num_batches = len(m.dataloader_test)
    m.model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in m.dataloader_test:
            X, y = X.to(m.device), y.to(m.device)
            pred = m.model(X)
            test_loss += m.loss(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print({"accuracy": 100 * correct, "loss": test_loss})
    m.history.append({"accuracy": 100 * correct, "loss": test_loss})


def experiment_train_reference(experiment: Experiment):
    for i in tqdm(range(experiment.epochs)):
        experiment.model_train_step(experiment.model_reference)
        experiment.model_test(experiment.model_reference)


def freeze(model: nn.Module):
    for i in model.parameters():
        i.requires_grad = False


def unfreeze(model, submodules: List[str]):
    for s in submodules:
        submodule = model.get_submodule(s)
        for i in submodule.parameters():
            i.requires_grad = True


def collect_and_apply(
    general: GeneralExperimentModel, copies: List[CopyExperimentModel]
):
    for m in copies:
        for submodule in m.submodules:
            sd = m.model.get_submodule(submodule).state_dict()
            general.model.get_submodule(submodule).load_state_dict(sd)

    for m in copies:
        m.model.load_state_dict(general.model.state_dict())

def collect_and_apply_best(
    general: GeneralExperimentModel, copies: List[CopyExperimentModel]
):
    def find_best_model(copies: List[CopyExperimentModel]) -> CopyExperimentModel:
        current = copies[0]
        for m in copies[1:]:
            if m.history[-1]["loss"] < current.history[-1]["loss"]:
                current = m
        return current



    # if len(general.history) == 0:
    best = find_best_model(copies)
    print(f"best: {best.submodules}")
    general.model.load_state_dict(best.model.state_dict())

    # for m in copies:
    #     for submodule in m.submodules:
    #         sd = m.model.get_submodule(submodule).state_dict()
    #         general.model.get_submodule(submodule).load_state_dict(sd)

    for m in copies:
        m.model.load_state_dict(general.model.state_dict())


def experiment_train_general(e: Experiment):
    # for epoch in tqdm(range(epochs), unit="epoch", total=epochs):
    for epoch in tqdm(range(e.epochs)):
        # for m in tqdm(models, unit="model", total=len(models)):
        for m in e.model_copies:
            freeze(m.model)
            unfreeze(m.model, m.submodules)
            for _ in range(m.epochs):
                e.model_train_step(m)
                e.model_test(m)

        collect_and_apply_best(e.model_general, e.model_copies)
        # collect_and_apply(e.model_general, e.model_copies)
        e.model_test(e.model_general)
