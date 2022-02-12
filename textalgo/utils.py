import argparse
from .hyperpyyaml import load_hyperpyyaml


def load_yaml(filename):
    with open(filename) as f:
        conf = load_hyperpyyaml(f)
    return conf


def parse_args_as_dict(parser, return_plain_args=False, args=None):
    args = parser.parse_args(args=args)
    args_dic = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        args_dic[group.title] = group_dict
    args_dic["main_args"] = args_dic["optional arguments"]
    del args_dic["optional arguments"]
    if return_plain_args:
        return args_dic, args
    return args_dic


def prepare_parser_from_dict(dic, parser=None):

    def standardized_entry_type(value):
        """If the default value is None, replace NoneType by str_int_float.
        If the default value is boolean, look for boolean strings."""
        if value is None:
            return str_int_float
        if isinstance(str2bool(value), bool):
            return str2bool_arg
        return type(value)

    if parser is None:
        parser = argparse.ArgumentParser()
    for k in dic.keys():
        group = parser.add_argument_group(k)
        for kk in dic[k].keys():
            entry_type = standardized_entry_type(dic[k][kk])
            group.add_argument("--" + kk, default=dic[k][kk], type=entry_type)
    return parser


def str_int_float(value):
    if isint(value):
        return int(value)
    if isfloat(value):
        return float(value)
    elif isinstance(value, str):
        return value


def str2bool(value):
    if not isinstance(value, str):
        return value
    if value.lower() in ("yes", "true", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "n", "0"):
        return False
    else:
        return value


def str2bool_arg(value):
    value = str2bool(value)
    if isinstance(value, bool):
        return value
    raise argparse.ArgumentTypeError("Boolean value expected.")


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False