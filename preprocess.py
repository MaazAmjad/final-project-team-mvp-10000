import argparse
import html
import os
import re
import sys
from collections import Counter
from itertools import islice

import spacy
from lxml import etree
from tqdm import tqdm

nlp=spacy.load('en', disable=['parser','ner','tagger'])
rgx = re.compile(r'\S')

def do_xml_parse(fps, tag, max_elements=None, progress_message=None):
    """ Parses cleaned up spacy-processed XML files """

    for fp in fps: 
        fp.seek(0)

        elements = enumerate(islice(etree.iterparse(fp, tag=tag), max_elements))
        for i, (event, elem) in elements:
            yield elem
            elem.clear()
            if progress_message and (i % 10 == 0): 
                print(progress_message.format(i), file=sys.stderr, end='\r')
        if progress_message: print(file=sys.stderr)


class Article(object):
    """ Represents a single article stored in an etree """
    def __init__(self, article):
        """ article is an etree extracted from an XML file """
        self.article = article 

    def get_anchors(self):
        """ return a list of the anchor elements """
        anchors = self.article.findall('.//a')
        return anchors

    def items(self):
        """ return the items from the etree """
        return self.article.items()       

    def get_links(self):
        """ extract info on links into a list of dictionaries """
        anchors = self.article.findall('.//a')
        links = []
        for a in anchors:
            links.append(dict(a.items()))
            links[-1]['text'] = a.text
        return links

    def get_spacy_text(self):
        """ extract the text from the spacy field and unescape it """
        return repeated_unescape("".join([x for x in self.article.find("spacy").itertext()])).split()

    def get_text(self):
        """ extract the text from the article if not in the spacy field """
        text = " ".join([x for x in self.article.itertext()])
        return text

    def get_title(self):
        """ extract the title """
        return self.article.get('title')


def repeated_unescape(line):
    new = line
    old = ''
    while old != new:
        old = new
        new = html.unescape(old)
    return new


def process(article_tree, text, lower=False):
    """ spacy tokenize, lemmatize and POS tag some text """
    if lower:
        doc = nlp(text.lower())
    else:
        doc = nlp(text)
    spacy_tree = etree.SubElement(article_tree, 'spacy')
    lemma_tree = etree.SubElement(article_tree, 'lemma')
    tag_tree = etree.SubElement(article_tree, 'tag')
    spacy_output = []
    tag_output = []
    lemma_output = []
    for token in doc:
        word = token.text
        lemma = token.lemma_
        tag = token.tag_
        if token.is_sent_start:
            spacy_output.append('-EOS-')
            tag_output.append('-EOS-')
            lemma_output.append('-EOS-')
        if rgx.search(word):
            spacy_output.append(word)
            tag_output.append(tag)
            lemma_output.append(lemma)

    spacy_output.append('-EOS-')
    tag_output.append('-EOS-')
    lemma_output.append('-EOS-')    
    spacy_tree.text = ' '.join(spacy_output)
    tag_tree.text = ' '.join(tag_output)
    lemma_tree.text = ' '.join(lemma_output)
    assert len(spacy_output) == len(tag_output) == len(lemma_output)

    return spacy_tree, tag_tree, lemma_tree, doc


def spacy_tokenize(article_tree, article):
    """
    spacy tokenize an article
    """
    text = repeated_unescape(article.get_text())
    doc = nlp(text)
    spacy_tree = etree.SubElement(article_tree, 'spacy')
    spacy_tree.text = ' '.join([x.text for x in doc])
    return article_tree

def save_links(article_tree, article):
    """
    extract the links in an article
    """
    anchors = article.get_anchors() 
    for a in anchors:
        # if there is no text in the link it didn't show up(?)
        if a.text is None: continue
        doc = nlp(repeated_unescape(a.text))
        text = ' '.join([x.text for x in doc])
        a.text = text
        anchor_tree = etree.SubElement(article_tree, 'a')
        anchor_tree.text = text
        for (k,v) in a.items(): anchor_tree.set(k,v)
    return article_tree


def process_articles(fp_ins, fp_out, features, start=None, end=None):
    fp_out.write(b'<articles>\n')
    
    for a in do_xml_parse(fp_ins, 'article'): 
        article = Article(a)

        article_tree = etree.Element('article')

        # set its attributes to be the same as the old attributes
        for (k,v) in article.items():
            article_tree.set(k,v)
        if 'tags' in features:
            article_text = repeated_unescape(article.get_text())
            spacy_tree, tag_tree, lemma_tree, doc = process(article_tree, article_text)
        if 'spacy' in features and 'tags' not in features:
            article_tree = spacy_tokenize(article_tree, article)
        if 'links' in features:
            article_tree = save_links(article_tree, article)
        if 'titles' in features:
            title_text = article.get_title()
            title_tree = etree.SubElement(article_tree, 'title')
            spacyT_tree, tagT_tree, lemmaT_tree, Tdoc = process(title_tree, title_text, lower=True)

        fp_out.write(etree.tostring(article_tree, pretty_print=True))
    
    fp_out.write(b'</articles>\n')
    return open(fp_out.name, "rb")

if __name__=='__main__':

    # python3 parser.py  --spacy --links --tags --titles --range 0 5 training.xml training_processed.xml
    
    parser = argparse.ArgumentParser(description="Convert Semeval XML files to more usable XML files")
    parser.add_argument("infile", help="an XML file with articles")
    parser.add_argument("outfile", help="an XML file with articles")
    parser.add_argument("--spacy", action="store_true")
    parser.add_argument("--links", action="store_true")
    parser.add_argument("--tags", action="store_true")
    parser.add_argument("--titles", action="store_true")
    parser.add_argument("--range", nargs=2, default=(None, None), help="article range for distributed processing")
    args = parser.parse_args()
    
    if args.tags:
        # Change nlp to do tagging, too...
        nlp=spacy.load('en')

    features = []
    for arg, feature in [(args.links, "links"), (args.spacy, "spacy"), (args.tags, "tags"), 
                    (args.titles, "titles")]:
        if arg:
            features.append(feature)

    assert len(features) > 0

    fp_in = open(args.infile, 'rb')
    if args.range == (None, None):
        (start, stop) = (None, None)
        outfile = args.outfile
    else:
        (start, stop) = [int(x) for x in args.range]
        outfile = "%s.%d_%d.xml" % (args.outfile[:-4], start, stop)
    fp_out = open(outfile, 'wb')
    read_articles(fp_in, fp_out, features, start, stop)
    fp_out.close()
