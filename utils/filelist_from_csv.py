import os
import pandas as pd
import argparse


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=str)
    parser.add_argument("--paths_column", type=int, default=0)
    parser.add_argument("--nr_of_labels", type=int, default=1)
    parser.add_argument("--labels_columns_start", type=int, default=1)
    parser.add_argument("path_out", type=str)
    return parser.parse_args()


def write_list(path_out, paths, labels):

    with open(path_out, "w") as fout:
        for i, (path, label) in enumerate(zip(paths, labels)):
            line = "%d\t" % i
            for j in label:
                line += "%f\t" % j
            line += "%s\n" % path
            fout.write(line)


def read_csv(args):
    df = pd.read_csv(os.path.expanduser(args.csv_path), header=None)
    paths = df.iloc[:, args.paths_column].values
    labels = df.iloc[
        :, args.labels_columns_start : args.labels_columns_start + args.nr_of_labels
    ].values
    return paths, labels


def main(args):
    paths, labels = read_csv(args)
    write_list(args.path_out, paths, labels)


if __name__ == "__main__":
    main(arg_parser())
