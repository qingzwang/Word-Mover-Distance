from itertools import product
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import euclidean, cosine
import pulp
import gensim
from nltk.corpus import stopwords
import argparse, json
from pathos.multiprocessing import ProcessingPool as Pool
import os

stop_words = stopwords.words('english')
wvmodel = gensim.models.KeyedVectors.load_word2vec_format(
    '../trained_models/word2vec/GoogleNews-vectors-negative300.bin', binary=True
)
# wvmodel = gensim.models.Word2Vec.load('../trained_models/word2vec/mscoco_all_cbow_negative_sampling.bin')
vocab = wvmodel.wv.vocab.keys()


def tokens_to_fracdict(tokens):
    cntdict = defaultdict(lambda : 0)
    for token in tokens:
        cntdict[token] += 1
    totalcnt = sum(cntdict.values())
    return {token: float(cnt)/totalcnt for token, cnt in cntdict.items()}


def word_mover_distance_probspec(first_sent_tokens, second_sent_tokens, wvmodel, lpFile=None):
    all_tokens = list(set(first_sent_tokens+second_sent_tokens))
    wordvecs = {token: wvmodel[token] for token in all_tokens}

    first_sent_buckets = tokens_to_fracdict(first_sent_tokens)
    second_sent_buckets = tokens_to_fracdict(second_sent_tokens)

    T = pulp.LpVariable.dicts('T_matrix', list(product(all_tokens, all_tokens)), lowBound=0)

    prob = pulp.LpProblem('WMD', sense=pulp.LpMinimize)
    prob += pulp.lpSum([T[token1, token2]*euclidean(wordvecs[token1], wordvecs[token2])
                        for token1, token2 in product(all_tokens, all_tokens)])
    for token2 in second_sent_buckets:
        prob += pulp.lpSum([T[token1, token2] for token1 in first_sent_buckets])==second_sent_buckets[token2]
    for token1 in first_sent_buckets:
        prob += pulp.lpSum([T[token1, token2] for token2 in second_sent_buckets]) == first_sent_buckets[token1]

    if lpFile != None:
        prob.writeLP(lpFile)

    prob.solve()

    return prob


def sentences2tokens(sentences):
    if (type(sentences) is str) or (type(sentences) is unicode):
        tokens = sentences.split(' ')
    elif type(sentences) is list:
        tokens = []
        for sentence in sentences:
            tokens.extend(sentence.split(' '))
    else:
        raise IOError
    filter_token = []
    for token in tokens:
        if (token not in stop_words) and (token in vocab):
            filter_token.append(token)
    # tokens = [token for token in tokens if token not in stop_words and token in vocab]
    return filter_token


def get_gt_tokens(coco_file='../data/files/dataset_coco.json', coco_tokens_file='../data/files/coco_tokens_Google_news.json'):
    if os.path.exists(coco_tokens_file):
        with open(coco_tokens_file, 'r') as f:
            dataset = json.load(f)
        return dataset

    print 'Processing ground-truth data...'
    with open(coco_file, 'r') as f:
        dataset = json.load(f)

    def f(images):
        image_tokens = {}
        # images = dataset['images']
        for image in images:
            sentence = image['sentences']
            image_id = str(image['cocoid'])
            tokens = []
            for s in sentence:
                tokens.extend(s['tokens'])
            filter_token = []
            for token in tokens:
                if (token not in stop_words) and (token in vocab):
                    filter_token.append(token)
            # tokens = [token for token in tokens if token not in stop_words and token in vocab]
            image_tokens[image_id] = filter_token
        return image_tokens

    all_images = dataset['images']
    num_images = len(all_images)
    num_workers = 30
    num_per_split = num_images // num_workers
    images_split = []
    for i in range(num_workers):
        if i == (num_workers - 1):
            images_split.append(all_images[(i * num_per_split):])
        else:
            images_split.append(all_images[(i*num_per_split):((i+1)*num_per_split)])
    pool = Pool(num_workers)
    all_images_tokens = pool.map(f, images_split)
    pool.close()
    pool.join()
    all_token_dict = {}
    for d in all_images_tokens:
        all_token_dict.update(d)
    with open(coco_tokens_file, 'w') as f:
        json.dump(all_token_dict, f)
    return all_token_dict


def get_generated_tokens(res_file, num=10):
    with open(res_file, 'r') as f:
        res = json.load(f)
    image_tokens = {}
    if 'merge_results' in res_file:
        print 'Processing merged results...'
        for im in res:
            image_id = str(im['image_id'])
            caption = im['captions'][0:num]
            tokens = sentences2tokens(caption)
            image_tokens[image_id] = tokens
    elif 'results_bs3' in res_file:
        print 'Processing beam search results...'
        for im in res:
            image_id = str(im['image_id'])
            caption = im['caption']
            tokens = sentences2tokens(caption)
            image_tokens[image_id] = tokens
    else:
        raise IOError
    return image_tokens


def word_mover_distance(first_sent_tokens, second_sent_tokens, wvmodel, lpFile=None, is_exp=1):
    prob = word_mover_distance_probspec(first_sent_tokens, second_sent_tokens, wvmodel, lpFile=lpFile)
    if is_exp == 1:
        return np.exp(-pulp.value(prob.objective))
    else:
        return pulp.value(prob.objective)


def compute_scores(args):
    results_file = args.results_file
    scores_file = args.score_file
    num_captions = args.num_captions
    is_exp = args.exp
    generated_image_tokens = get_generated_tokens(res_file=results_file, num=num_captions)
    gt_image_tokens = get_gt_tokens()
    print('number of test images: %d, all images: %d')%(len(generated_image_tokens), len(gt_image_tokens))
    all_image_ids = generated_image_tokens.keys()

    def f(image_id_thread):
        image_ids = image_id_thread[0]
        thread_num = image_id_thread[1]
        scores = {}
        for image_id in image_ids:
            res_tokens = generated_image_tokens[str(image_id)]
            gt_tokens = gt_image_tokens[str(image_id)]
            wmd_score = word_mover_distance(res_tokens, gt_tokens, wvmodel=wvmodel, is_exp=is_exp)
            scores[image_id] = wmd_score
            print('Thread: %d, Image ID: %s, WMD score: %.5f')%(thread_num, image_id, wmd_score)
        return scores

    num_images = len(all_image_ids)
    num_workers = 20
    num_per_split = num_images // num_workers
    images_split = []
    for i in range(num_workers):
        if i == (num_workers - 1):
            images_split.append([all_image_ids[(i * num_per_split):], i])
        else:
            images_split.append([all_image_ids[(i * num_per_split):((i + 1) * num_per_split)], i])
    pool = Pool(num_workers)
    all_scores = pool.map(f, images_split)
    pool.close()
    pool.join()
    scores = {}
    for s in all_scores:
        scores.update(s)
    with open(scores_file, 'w') as f:
        json.dump(scores, f)
    total_score = 0
    for key in scores.keys():
        total_score += scores[key]
    print('WMD score: %.5f')%(total_score / len(scores))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--results_file', type=str, help='your results file, the name should be results_bs3.json or merge_results.json')
    args.add_argument('--score_file', type=str, help='the json file that is used to store the metric scores')
    args.add_argument('--num_captions', type=int, default=10, help='how many captions for each image')
    args.add_argument('--exp', type=int, default=1, help='whether using negative exponential, 1 for using, otherwise, not use ')
    args = args.parse_args()
    compute_scores(args)
