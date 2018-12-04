import argparse
import sys
from lxml import etree
from itertools import islice
from html import unescape

#
# extract_articles.py: Extract the text from articles 
#

#
# Helpers
#

def do_xml_parse(fp, tag, max_elements=None, progress_message=None):
    """
    Parses cleaned up spacy-processed XML files. This is done by having
    the function be a generator so that it never has to store the entire
    data in memory.
    """
    fp.seek(0)

    # Constructs an iterator that only includes at most max_elements from the iterator
    # etree.iterparse along with the current index.
    elements = enumerate(islice(etree.iterparse(fp, tag=tag), max_elements))
    for i, (event, elem) in elements:
        yield elem   # Returns the current element.
        elem.clear() # Empties out the list contained in elem to save memory.
        if progress_message and (i % 1000 == 0):
            print(progress_message.format(i), file=sys.stderr, end='\r')
    if progress_message: print(file=sys.stderr)

def extract_text2(article):
    return unescape("".join([x for x in article.find("spacy").itertext()]).lower())

#
# CLI
#

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=argparse.FileType('rb'), help='input xml file')
    parser.add_argument('output_file', type=argparse.FileType('w'), help='output file')
    parser.add_argument('--max_articles', type=int, default=-1, help='maximum number of articles to process')
    parser.add_argument('--gt', action='store_true', help='whether your input is a ground truth file and you seek to have the article classifications')

    args = parser.parse_args()
    
    max_len = -1
    for index, article in enumerate(do_xml_parse(args.input_file, 'article')):
        if args.max_articles == index:
            break

        if index % 1000 == 0:
            print(index, end='\r')
        
        if args.gt:
            article_text = article.get('hyperpartisan')
        else:
            article_text = " ".join(extract_text2(article).split())
            if len(article_text.split(" ")) > max_len:
                max_len = len(article_text.split(" "))

        args.output_file.write(article_text + "\n")
    
    print("Longest length:", max_len)
    args.input_file.close()
    args.output_file.close()
