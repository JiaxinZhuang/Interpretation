"""Autoencoder.
"""

import config
from utils.function import init_logging, init_environment, get_lr, \
    freeze_model, timethis


@timethis
def main():
    configs = config.Config()
    configs_dict = configs.get_config()
    exp = configs_dict["experiment_index"]
    cuda_id = configs_dict["cuda"]
    num_workers = configs_dict["num_workers"]
    seed = configs_dict["seed"]
    n_epochs = configs_dict["n_epochs"]
    log_dir = configs_dict["log_dir"]
    model_dir = configs_dict["model_dir"]
    batch_size = configs_dict["batch_size"]
    learning_rate = configs_dict["learning_rate"]
    dataset_name = configs_dict["dataset"]
    re_size = configs_dict["re_size"]
    input_size = configs_dict["input_size"]
    backbone = configs_dict["backbone"]
    eval_frequency = configs_dict["eval_frequency"]
    resume = configs_dict["resume"]
    optimizer = configs_dict["optimizer"]
    warmup_epochs = configs_dict["warmup_epochs"]
    initialization = configs_dict["initialization"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _print = init_logging(log_dir, exp).info
    configs.print_config(_print)
    init_environment(seed=seed, cuda_id=cuda_id, _print=_print)
    tf_log = os.path.join(log_dir, exp)
    writer = SummaryWriter(log_dir=tf_log)

    if dataset_name == "DTD":

