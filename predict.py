#!/usr/bin/env python3

import argparse
import numpy as np
import os
import preprocess
import run_classifier

def do_preprocess(fp_ins, fp_out): 
    ## Set these depending on which attributes you want to keep -- no need to generate things you're not using!
    features = ["spacy"] # links, tags, titles are the other options
    return preprocess.process_articles(fp_ins, fp_out, features)

def do_predict(args): 

    # Preprocessing! (This takes a while...)
    input_dir_files = os.listdir(args.inputDataset)
    input_file_handles = [open(os.path.join(args.inputDataset, x), "rb") for x in input_dir_files]
    temp_dir = args.inputDataset.rstrip("/") + "_preprocessed"
    if not(os.path.exists(temp_dir)): 
        os.mkdir(temp_dir)
    temp_fname = os.path.join(temp_dir, "articles.xml")
    temp_fp = open(temp_fname, "wb")
    temp_fp = do_preprocess(input_file_handles, temp_fp)

    ## Read in all the important model things
    feature_maker = pickle.load(args.model)
    label_maker = pickle.load(args.model)
    clf=pickle.load(args.model)

    ## Get features for articles to predict on
    X_test, ids_test = feature_maker.process(temp_fp, max_instances=None)

    ## Predict! 
    class_probs = clf.predict_proba(X_test)
    y_pred=np.argmax(class_probs,axis=1)

    outfile = os.path.join(args.outputDir, "predictions.txt")
    with open(outfile, "w") as fp:
        for article_id, pred, class_confidence in zip(ids_test, y_pred, class_probs):
            print(article_id, end=" ", file=fp)
            print(label_maker[pred], end=" ", file=fp)
            print("{:.2f}".format(class_confidence[pred]), file=fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("modelPath", help="Saved model")
    parser.add_argument("inputDataset", help="Directory with the xml file(s) to test")
    parser.add_argument("outputDir", help="Directory to put the predictions.txt file in")

    args = parser.parse_args()
    do_predict(args)

