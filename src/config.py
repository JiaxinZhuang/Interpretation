"""Config.
All hyper paramerers and path can be changed in this file
"""

import argparse
from utils.function import str2bool, str2list

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
        self.config = dict()

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
        self.parser.add_argument('--experiment_index', default="None", type=str,
                                 help="001, 002, ...")
        self.parser.add_argument('--cuda', default='0',
                                 help="cuda visible device")
        self.parser.add_argument("--num_workers", default=2, type=int,
                                 help="num_workers of dataloader")
        self.parser.add_argument('--dataset', default="CUB", type=str,
                                 choices=["CUB", "TinyImageNet", "stl10", \
                                          "mnist"], \
                                 help="dataset name")
        # log related
        self.parser.add_argument('--log_dir', default="./saved/logdirs/",
                                 type=str, help='store tensorboard files, \
                                 None means not to store')
        self.parser.add_argument('--model_dir', default="./saved/models/",
                                 type=str, help='store models, ../saved/models')
        self.parser.add_argument('--generated_dir', default="./saved/generated/",
                                 type=str, help='store models, ../saved/generated')

        self.parser.add_argument('--learning_rate', default=0.01, type=float,
                                 help="lr")
        self.parser.add_argument("--batch_size", default=64, type=int,
                                 help="batch size of each epoch, for test only")
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
                                 help="weight decay for optimizer")

        self.parser.add_argument("--optimizer", default="SGD", type=str,
                                 choices = ["SGD", "Adam"],
                                 help="use SGD or Adam")
        self.parser.add_argument("--input_size", default=224, type=int,
                                 help="image input size for model")
        self.parser.add_argument("--backbone", default="resnet34", type=str,
                                 help="backbone for model")

        self.parser.add_argument("--re_size", default=256, type=int,
                                 help="resize to the size")

    def _add_customized_setting(self):
        """Add customized setting
        """
        self.parser.add_argument('--selected_layer', default=5, type=int,
                help='For convNet: 3, 5')
        self.parser.add_argument('--selected_filter', default=25, type=int,
                help='For convNet: 3(max 20), 5(max 50)')
        self.parser.add_argument("--alpha", default=0.5, type=float,
                                 help="0-1")
        self.parser.add_argument("--class_index", default="5", type=str2list,
                                 help="[0,1,2,3,4,5,6,7] ... ")
        self.parser.add_argument("--num_class", default=1, type=int,
                                 help="no more than numbers of each class")
        #self.parser.add_argument('--normalize', default=True, type=str2bool,
        #                         help='128, 256, 512, 1024, 2048')
        #self.parser.add_argument('--margin', default=0.2, type=float,
        #                         help='margin for triplet loss')
        #self.parser.add_argument('--K', default=3, type=int,
        #                         help="postive dot for training triplets")
        #self.parser.add_argument('--M', default=3, type=int,
        #                         help='negative dot for testing triplets')
        #self.parser.add_argument('--k_predict', default=3, type=int,
        #                         help='k used for predict phase')
        #self.parser.add_argument('--batch_n_classes', default=10, type=int,
        #                         help='depend on your dataset')
        #self.parser.add_argument('--batch_n_class_num', default=20, type=int,
        #                         help='depend on your dataset, \
        #                         number for each class per batch')
        #self.parser.add_argument("--metric_function", default="l2", type=str,\
        #                         help="metric function used for \
        #                         pairwise_distance, l2 or cos")
        self.parser.add_argument("--server", default="local", type=str,
                                 help="server to run the code")
        #self.parser.add_argument("--bilinear", default=True, type=str2bool,
        #                         help="use bilinear within model")
        #self.parser.add_argument("--finetune", default="", type=str,
        #                         help=" \"\" or 2001\"40")
        #self.parser.add_argument("--freeze", default=True, type=str2bool,
        #                         help="freeze embedding parameter when ")
        #self.parser.add_argument("--save_memory", default=False, type=str2bool,
        #                         help="Save memory when calculate pairdistance ")
        #self.parser.add_argument("--triplet_method", default="batch_all",
        #                         type=str, help="batch_all, batch_hard")
        #self.parser.add_argument("--use_sampler", default=False,
        #                         type=str2bool, help="False or True")

    def _load_common_setting(self):
        """Load default setting from Parser
        """
        self.config['experiment_index'] = self.args.experiment_index
        self.config['cuda'] = self.args.cuda
        self.config["num_workers"] = self.args.num_workers

        self.config['dataset'] = self.args.dataset

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

        self.config["input_size"] = self.args.input_size
        self.config["backbone"] = self.args.backbone
        self.config["re_size"] = self.args.re_size

        self.config["optimizer"] = self.args.optimizer

    def _load_customized_setting(self):
        """Load sepcial setting
        """
        self.config["selected_filter"] = self.args.selected_filter
        self.config["selected_layer"] = self.args.selected_layer
        self.config["alpha"] = self.args.alpha
        self.config["class_index"] = self.args.class_index
        self.config["num_class"] = self.args.num_class
        #self.config['embedding_len'] = self.args.embedding_len
        #self.config["normalize"] = self.args.normalize
        #self.config['margin'] = self.args.margin
        #self.config["M"] = self.args.M
        #self.config["K"] = self.args.K
        #self.config["k_predict"] = self.args.k_predict
        #self.config['batch_n_class_num'] = self.args.batch_n_class_num
        #self.config['batch_n_classes'] = self.args.batch_n_classes
        #self.config["metric_function"] = self.args.metric_function
        self.config["server"] = self.args.server
        #self.config["bilinear"] = self.args.bilinear
        #self.config["finetune"] = self.args.finetune
        #self.config["freeze"] = self.args.freeze
        #self.config["save_memory"] = self.args.save_memory
        #self.config["triplet_method"] = self.args.triplet_method
        pass

    #def _modify_config(self):
    #    """Modify some config
    #    """
    #    # freeze only used when finetune
    #    if self.config["finetune"] == False:
    #        self.args.freeze = False

    def _path_suitable_for_server(self):
        """Path suitable for server
        """
        if self.config["server"] == "desktop":
            self.config["log_dir"] = "/home/lincolnzjx/Desktop/Interpretation/saved/logdirs"
            self.config["model_dir"] = "/home/lincolnzjx/Desktop/Interpretation/saved/models"
            self.config["generated_dir"] = "/home/lincolnzjx/Desktop/Interpretation/saved/generated"
        if self.config["server"] == "local":
            self.config["log_dir"] = "/media/lincolnzjx/Disk21/interpretation/saved/logdirs"
            self.config["model_dir"] = "/media/lincolnzjx/Disk21/interpretation/saved/models"
            self.config["generated_dir"] = "/media/lincolnzjx/Disk21/Interpretation/saved/generated"
        elif self.config["server"] == "ls15":
            self.config["log_dir"] = "/data15/jiaxin/Fine-Grained-Recognition/saved/logdirs"
            self.config["model_dir"] = "/data15/jiaxin/Fine-Grained-Recognition/saved/models"
            self.config["generated_dir"] = "/data15/jiaxin/Fine-Grained-Recognition/saved/generated"
        elif self.config["server"] == "ls16":
            self.config["log_dir"] = "/data16/jiaxin/Fine-Grained-Recognition/saved/logdirs"
            self.config["model_dir"] = "/data16/jiaxin/Fine-Grained-Recognition/saved/models"
            self.config["generated_dir"] = "/data16/jiaxin/Fine-Grained-Recognition/saved/generated"
        elif self.config["server"] == "lab_center":
            self.config["log_dir"] = "/home/jiaxin/Fine-Grained-Recognition/saved/logdirs"
            self.config["model_dir"] = "/home/jiaxin/Fine-Grained-Recognition/saved/models"
            self.config["generated_dir"] = "/home/jiaxin/Fine-Grained-Recognition/saved/generated"

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
