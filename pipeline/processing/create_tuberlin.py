"""
This file is used to transform the training set to a CSV file.
This allows analysis in Excel.
"""
import argparse
import os
import pickle

import pandas as pd


def setup_create_tuberlin(parser: argparse.ArgumentParser, group):
    group.add_argument('--create_tuberlin', action="store_true")

    def create_tuberlin(args, device):
            if not args.create_tuberlin:
                return

            # TODO
            classes = [name for name in os.listdir(args.source)]
            #print(list(filter(lambda x: os.path.isdir(os.path.join(args.source, x)), classes)))
            #return


            df = pd.DataFrame(index=range(0, 20000), columns=["cl", "filename"])
            pos = 0
            for cl in classes:
                cl_path = os.path.join(args.source, cl)
                if not os.path.isdir(cl_path):
                    continue

                for file in os.listdir(cl_path):
                    if os.path.isfile(os.path.join(cl_path, file)):
                        df.iloc[pos]['cl'] = cl
                        df.iloc[pos]['filename'] = os.path.join(cl, file)
                        pos += 1

            df = df.sample(frac=1).reset_index(drop=True)
            print(df)
            train_df = df[0:int(20000/100*80)]
            #train_df = df[0:100]
            train_df.reset_index(drop=True)
            validation_df = df[int(20000/100*80):]
            #validation_df = df[100:200]
            validation_df.reset_index(drop=True)

            train_target_file_path = os.path.join(args.source, 'train_set.p')
            validation_target_file_path = os.path.join(args.source, 'validation_set.p')
            if os.path.exists(train_target_file_path):
                os.remove(train_target_file_path)
            if os.path.exists(validation_target_file_path):
                os.remove(validation_target_file_path)

            pickle.dump(train_df, open(train_target_file_path, "wb"))
            pickle.dump(validation_df, open(validation_target_file_path, "wb"))

            print("Finished!")

    return create_tuberlin
