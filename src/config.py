"""Config.
All hyper paramerers and path can be changed in this file
"""

import sys
import argparse
from utils.function import str2list, str2bool


class Config:
    """Config
    Attributes:
        parser: to read all config
        args: argument from argument parser
        config: save config in pairs like key:value
    """
    def __init__(self):
        """Load common and customized settings
        """
        super(Config, self).__init__()
        self.parser = argparse.ArgumentParser(description='Interpretation')
        self.config = {}

        # add setting via parser
        self._add_common_setting()
        self._add_customized_setting()
        # get argument parser
        self.args = self.parser.parse_args()
        # load them into config
        self._load_common_setting()
        self._load_customized_setting()
        # load config for specific server
        self._path_suitable_for_server()

    def _add_common_setting(self):
        # need defined each time
        self.parser.add_argument('--experiment_index', default="None",
                                 type=str,
                                 help="001, 002, ...")
        self.parser.add_argument('--cuda', default=0, type=int,
                                 help="cuda visible device")
        self.parser.add_argument("--num_workers", default=2, type=int,
                                 help="num_workers of dataloader")
        self.parser.add_argument('--dataset', default="Caltech101", type=str,
                                 choices=["CUB", "TinyImageNet", "stl10",
                                          "mnist", "Caltech101", "ImageNet"],
                                 help="dataset name")
        self.parser.add_argument("--data_dir", default=None, type=str,
                                 help="data directory.")
        # log related
        self.parser.add_argument('--log_dir', default="./saved/logdirs/",
                                 type=str, help='store tensorboard files, \
                                 None means not to store')
        self.parser.add_argument('--model_dir', default="./saved/models/",
                                 type=str,
                                 help='store models, ../saved/models')
        self.parser.add_argument('--generated_dir',
                                 default="./saved/generated/",
                                 type=str,
                                 help='store models, ../saved/generated')

        self.parser.add_argument('--learning_rate', default=0.01, type=float,
                                 help="lr")
        self.parser.add_argument("--batch_size", default=64, type=int,
                                 help="batch size of each epoch, \
                                       for test only")
        self.parser.add_argument('--resume', default="001-215", type=str,
                                 help="resume exp and epoch")
        self.parser.add_argument("--n_epochs", default=1000000, type=int,
                                 help="n epochs to train")

        self.parser.add_argument("--eval_frequency", default=1, type=int,
                                 help="Eval train and test frequency")

        self.parser.add_argument('--seed', default=47, type=int,
                                 help="Random seed for pytorch and Numpy ")
        self.parser.add_argument('--eps', default=1e-7, type=float,
                                 help="episilon for many formulation")
        self.parser.add_argument("--weight_decay", default=1e-4, type=float,
                                 metavar="W",
                                 help="weight decay for optimizer")
        self.parser.add_argument("--momentum", default=0.9, type=float,
                                 metavar="M", help="momentum.")

        self.parser.add_argument("--optimizer", default="SGD", type=str,
                                 choices=["SGD", "Adam"],
                                 help="use SGD or Adam")
        self.parser.add_argument("--input_size", default=224, type=int,
                                 help="image input size for model")
        self.parser.add_argument("--re_size", default=256, type=int,
                                 help="resize to the size")
        self.parser.add_argument("--backbone", default="resnet34", type=str,
                                 help="backbone for model")
        self.parser.add_argument("--warmup_epochs", default=-1, type=int,
                                 help="epochs to use warm up")
        self.parser.add_argument("--initialization", default="default",
                                 type=str,
                                 choices=["xavier_normal", "default",
                                          "pretrained", "kaiming_normal",
                                          "kaiming_uniform", "xavier_uniform"],
                                 help="initializatoin method")

        self.parser.add_argument("--prof", dest="prof", action="store_true",
                                 help="Only run 10 iterations for profiling.")
        self.parser.add_argument("--print-freq", default=10, type=int,
                                 metavar="N",
                                 help="print frequency (default: 10)")

    def _add_customized_setting(self):
        """Add customized setting
        """
        self.parser.add_argument('--selected_layer', default=5, type=int,
                                 help='For convNet: 3, 5')
        self.parser.add_argument('--selected_filter', default=25, type=int,
                                 help='For convNet: 3(max 20), 5(max 50)')
        self.parser.add_argument("--alpha", default=0.5, type=float,
                                 help="0-1")
        self.parser.add_argument("--beta", default=1, type=float,
                                 help="PLEASE DO NOT CHANGE unless you are \
                                       VERY sure.")
        self.parser.add_argument("--gamma", default=0, type=float,
                                 help="gamma for regularization, default not \
                                       to use.")
        self.parser.add_argument("--class_index", default="5", type=str2list,
                                 help="[0,1,2,3,4,5,6,7] ... ")
        self.parser.add_argument("--num_class", default=1, type=int,
                                 help="no more than numbers of each class")
        self.parser.add_argument("--mode", default="keep", type=str,
                                 choices=["keep", "remove"],
                                 help="mode for loss function")
        self.parser.add_argument("--dropout", default=True, type=str2bool,
                                 choices=[True, False], help="Whether to use \
                                 dropout when training baseline")
        self.parser.add_argument("--clip_grad", default=False, type=str2bool,
                                 help="Whether to clip gradient")
        self.parser.add_argument("--inter", default=False, type=str2bool,
                                 choices=[True, False],
                                 help="interact between different channles")
        self.parser.add_argument("--rho", default=0, type=float,
                                 help="rho for interact between different \
                                       channles")
        self.parser.add_argument("--conv_bias", default=True, type=str2bool,
                                 choices=[True, False],
                                 help="whether to keep conv bias")
        self.parser.add_argument("--linear_bias", default=True, type=str2bool,
                                 choices=[True, False],
                                 help="whether to keep linear bias")
        self.parser.add_argument("--regularization", default="None", type=str,
                                 choices=["None", "L1", "L2",
                                          "ClipNorm", "ClipContribution"],
                                 help="Use which regularization method to \
                                       reduce the complexity of processed \
                                       images")
        self.parser.add_argument("--smoothing", default="None",
                                 choices=["None", "TotalVariation"],
                                 type=str, help="smoothing op, \
                                 default not to use.")
        self.parser.add_argument("--delta", default=0, type=float,
                                 help="coefficient for smoothing term.")
        self.parser.add_argument("--regular_ex", default=1, type=float,
                                 help="When using TotalVariation, exponential\
                                       for regularization.")
        self.parser.add_argument("--img_index", default=0, type=int,
                                 help="Img index to use. Batch_size has to be\
                                 1 when to be used.")
        self.parser.add_argument("--server", default="local", type=str,
                                 choices=["local", "ls15", "ls16", "ls31",
                                          "ls97", "desktop"],
                                 help="server to run the code")
        self.parser.add_argument("--rescale", type=str2bool, default=False,
                                 help="whether to rescale for recreate \
                                 images.")
        self.parser.add_argument("--freeze", type=str2bool, default=False,
                                 help="freeze parameter when training \
                                 beseline.")
        self.parser.add_argument("--dali", type=str2bool, default=False,
                                 help="dali to accelebrate load data.")
        self.parser.add_argument("--save_predict", type=str2bool,
                                 default=False, help="whether to save predict")

        self.parser.add_argument("--local_rank", default=0, type=int)
        self.parser.add_argument("--world_size", default=1, type=int)
        self.parser.add_argument("--distributed", default=False, type=str2bool,
                                 help="whether to use distribute.")
        self.parser.add_argument('--dist_url',
                                 default='tcp://127.0.0.1:23456',
                                 type=str, help='url used to set up distributed\
                                 training')

        self.parser.add_argument("--guidedReLU", type=str2bool,
                                 default=False, help="whether to use guild.")

    def _load_common_setting(self):
        """Load default setting from Parser
        """
        self.config['experiment_index'] = self.args.experiment_index
        self.config['cuda'] = self.args.cuda
        self.config["num_workers"] = self.args.num_workers

        self.config['dataset'] = self.args.dataset
        self.config["data_dir"] = self.args.data_dir

        self.config["resume"] = self.args.resume
        self.config['n_epochs'] = self.args.n_epochs

        self.config['learning_rate'] = self.args.learning_rate
        self.config['batch_size'] = self.args.batch_size

        self.config['seed'] = self.args.seed

        self.config["eval_frequency"] = self.args.eval_frequency
        self.config['log_dir'] = self.args.log_dir
        self.config['model_dir'] = self.args.model_dir
        self.config['generated_dir'] = self.args.generated_dir

        self.config["eps"] = self.args.eps
        self.config["weight_decay"] = self.args.weight_decay
        self.config["momentum"] = self.args.momentum

        self.config["input_size"] = self.args.input_size
        self.config["backbone"] = self.args.backbone
        self.config["re_size"] = self.args.re_size

        self.config["optimizer"] = self.args.optimizer
        self.config["warmup_epochs"] = self.args.warmup_epochs
        self.config["initialization"] = self.args.initialization

        self.config["prof"] = self.args.prof
        self.config["print_freq"] = self.args.print_freq

    def _load_customized_setting(self):
        """Load sepcial setting
        """
        self.config["selected_filter"] = self.args.selected_filter
        self.config["selected_layer"] = self.args.selected_layer
        self.config["alpha"] = self.args.alpha
        self.config["beta"] = self.args.beta
        self.config["gamma"] = self.args.gamma
        self.config["class_index"] = self.args.class_index
        self.config["num_class"] = self.args.num_class
        self.config["mode"] = self.args.mode
        self.config["dropout"] = self.args.dropout
        self.config["clip_grad"] = self.args.clip_grad
        self.config["inter"] = self.args.inter
        self.config["rho"] = self.args.rho
        self.config["conv_bias"] = self.args.conv_bias
        self.config["linear_bias"] = self.args.linear_bias
        self.config["regularization"] = self.args.regularization
        self.config["smoothing"] = self.args.smoothing
        self.config["delta"] = self.args.delta
        self.config["regular_ex"] = self.args.regular_ex
        self.config["img_index"] = self.args.img_index
        self.config["rescale"] = self.args.rescale
        self.config["server"] = self.args.server
        self.config["freeze"] = self.args.freeze
        self.config["dali"] = self.args.dali
        self.config["save_predict"] = self.args.save_predict

        self.config["local_rank"] = self.args.local_rank
        self.config["world_size"] = self.args.world_size
        self.config["distributed"] = self.args.distributed
        self.config["dist_url"] = self.args.dist_url

        self.config["guidedReLU"] = self.args.guidedReLU

    def _path_suitable_for_server(self):
        """Path suitable for server
        """
        if self.config["server"] in ["desktop", "ls15", "ls16", "ls31",
                                     "ls97", "lab_center"]:
            self.config["log_dir"] = "./saved/logdirs"
            self.config["model_dir"] = "./saved/models"
            self.config["generated_dir"] = "./saved/generated"
        else:
            print("Illegal server configuration")
            sys.exit(-1)

        if self.config["server"] == "ls15":
            self.config["data_dir"] = "/data15/Public/Datasets/"
        elif self.config["server"] == "ls16":
            self.config["data_dir"] = "/data16/Public/Datasets/"
        elif self.config["server"] == "ls97":
            self.config["data_dir"] = "/data/Public/Datasets/"
        elif self.config["server"] == "desktop":
            self.config["data_dir"] = "/media/lincolnzjx/HardDisk/Datasets/"
        else:
            print("Illegal data_dir")
            sys.exit(-1)

    def print_config(self, _print=None):
        """print config
        """
        _print("==================== basic setting start ====================")
        for arg in self.config:
            _print('{:20}: {}'.format(arg, self.config[arg]))
        _print("==================== basic setting end ====================")

    def get_config(self):
        """return config
        """
        return self.config
