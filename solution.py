import numpy as np
import sys, datetime, logging
import os
from os.path import isfile, join
import pickle

def setup_logger(name):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('log/{}.log'.format(now), mode='w')
    handler.setFormatter(formatter)

    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    
    return logger

def report_progress(progress, total, lbar_prefix = '', rbar_prefix=''):
    percent = round(progress / float(total) * 100, 2)
    buf = "{0}|{1}| {2}{3}/{4} {5}%".format(lbar_prefix, ('#' * round(percent)).ljust(100, '-'),
        rbar_prefix, progress, total, percent)
    sys.stdout.write(buf)
    sys.stdout.write('\r')
    sys.stdout.flush()

    
######################################################
#                      Solution                      #
######################################################


logger = setup_logger("Solution")

def load_input(file_name):
    load_name = file_name + ".txt"
    with open(join(file_name, load_name), mode="r", encoding="ascii") as file:
        # Load first line
        num_images = file.readline()
        H_images, V_images = dict(), dict()
        
        index = 0
        for row in file.readlines():
            info = row.split()
            if len(info) <= 2:
                continue
            shape, num_tags, tags = info[0], info[1], info[2:]
            assert len(tags) == int(num_tags)

            if shape == "H":
                H_images[index] = np.array(tags)
            elif shape == "V":
                V_images[index] = np.array(tags)
            
            index += 1
            if index >= 2000:
                break
            
    return num_images, H_images, V_images


def tag_interest_V(tags1, tags2):
    """
    Inputs: 
        tags1: np.array
        tags2: np.array
    """
    return np.unique(np.concatenate((tags1, tags2), axis=0)).size


def best_pairs_V(matrix):
    best_pairs = list()
    visited = set()
    for i in range(matrix.shape[0]):
        if i in visited:
            continue
        visited.add(i)
        if np.max(matrix[i]) > 0:
            j = np.argmax(matrix[i])
            best_pairs.append(tuple([i, j]))
            visited.add(j+1)
            matrix[:, j] = 0
    return best_pairs
         
    
def process_V(V_images):
    num_images = len(V_images)
    if num_images == 0:
        return []
    
    # Index Mapping
    v_keys = list(V_images.keys())
    index = dict()
    for i in range(len(v_keys)):
        index[i] = v_keys[i]
    
    load_name = "v_" + file_name + ".gz"
    if os.path.isfile(join(file_name, load_name)):
        logger.info("Loading V Matrix")
        with open(join(file_name, load_name), "rb") as file:
            matrix = pickle.load(file)
        logger.info("Loaded V Matrix [dim={}]".format(matrix.shape[0]))
    else: 
        matrix = np.full((num_images-1, num_images-1), -1)
    
        for i in range(num_images-1):
            for j in range(i+1, num_images-1):
                merged_tags = tag_interest_V(V_images[index[i]], V_images[index[j]])
                matrix[i][j] = merged_tags
        
        logger.info("Storing V Matrix")
        with open(join(file_name, load_name), "wb") as file:
            pickle.dump(matrix, file)
    
    best_pairs = best_pairs_V(matrix)
    best_pairs = [(index[x], index[y]) for x, y in best_pairs]
    return best_pairs


def calc_interest(tags1, tags2):
    """
    Input:
        tags1: np.array
        tags2: np.array
    """
    intersect = np.intersect1d(tags1, tags2).size
    diff1 = np.setdiff1d(tags1, tags2).size
    diff2 = np.setdiff1d(tags2, tags1).size
    return min(intersect, diff1, diff2)
    

def best_sequence_all(matrix):
    
    logger.info("Computing Sequence")
    visited = set()
    best_sequence = []
    total_interest = 0
    
    row = 0
    count = 1
    while count <= matrix.shape[0]:
        if row in visited:
            continue
        visited.add(row)
        best_sequence.append(row)
        max_interest = np.max(matrix[row])
        if max_interest < 0:
            break
        col = np.argmax(matrix[row])
        total_interest += int(max_interest)
        
        matrix[:, row] = -1
        row = col
        report_progress(count, matrix.shape[0])
        count += 1
    return best_sequence, total_interest
            

def process_all(H_images, V_images, V_merged, file_name):
    """
    Inputs:
        H_images: dict() -> [key:23 - value:tags(np.array)]
        V_images: dict() -> [key:24 - value:tags(np.array)]
        V_merged: list() -> [(0, 33), (15, 88), ...]
    """
    
    # Index mapping & Tag index mapping
    h_keys = list(H_images.keys())
    index = dict()
    tags = dict()
    for i in range(len(h_keys)):
        index[i] = h_keys[i]
        tags[i] = H_images[h_keys[i]]

    for i in range(len(V_merged)):
        index[i+len(h_keys)] = V_merged[i]
        tags[i+len(h_keys)] = np.unique(np.concatenate([V_images[k] for k in V_merged[i]], axis=0))

    load_name = "matrix_" + file_name + ".gz"
    if os.path.isfile(join(file_name, load_name)):
        logger.info("Loading Matrix")
        with open(join(file_name, load_name), "rb") as file:
            matrix = pickle.load(file)
        logger.info("Loaded Matrix [dim={}]".format(matrix.shape[0]))
    else:

        # Construct Matrix 
        # TODO - Simplify the process
        num_images = len(index)
        logger.info("Constructing Matrix [dim={}]".format(num_images))

        matrix = np.full((num_images, num_images), -1)
        for i in range(num_images):
            for j in range(num_images):
                if i == j:
                    continue
                matrix[i][j] = calc_interest(tags[i], tags[j])
                
        logger.info("Storing Matrix")
        with open(join(file_name, load_name), "wb") as file:
            pickle.dump(matrix, file)
    

    best_sequence, total_interest = best_sequence_all(matrix)
    best_sequence = [index[i] for i in best_sequence]
    return best_sequence, total_interest
    

def main(file_name):
    num_images, H_images, V_images = load_input(file_name)
    
    logger.info("V_merged Start")
    V_merged = process_V(V_images) # [(0, 33), (15, 88), ...]
    logger.info("V_merged Complete")

    # Construct big matrix
    logger.info("Process_All Start")
    best_sequence, total_interest = process_all(H_images, V_images, V_merged, file_name)
    
    logger.info("Best Sequence has length: {}".format(len(best_sequence)))
    logger.info("Achieve Interest: {}".format(total_interest))
    
    logger.info("Saving Output")
    content = []
    content.append(str(len(best_sequence)) + "\n")
    visited = []
    for index in best_sequence:
        if isinstance(index, tuple):
            for i in index:
                if i in visited:
                    break
            for i in index:
                visited.append(i)
            index = [str(i) for i in index]
            element = " ".join(index)
            content.append(str(element) + "\n")
        else:
            if index in visited:
                continue
            content.append(str(index) + "\n")
            visited.append(index)
    
    output_name = "output_" + file_name + ".txt"
    with open(join(file_name, output_name), "w", encoding="ascii") as file:
        file.writelines(content)
    
    
############################################################################

file_name = "c_memorable_moments"
main(file_name)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# END