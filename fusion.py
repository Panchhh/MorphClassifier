import itertools
from os.path import join
from itertools import zip_longest
from score import _save_file


def score_fusion(file_path_list, output_path):
    genuine, impostor = [], []
    size = len(file_path_list)
    for file_path in file_path_list:
        genuine_path = join(file_path, 'SVC_bonafide.score')
        impostor_path = join(file_path, 'SVC_impostor.score')
        with open(genuine_path) as f:
            genuine = [sum(n) for n in zip_longest(genuine, map(float, f.read().splitlines()), fillvalue=0)]
        with open(impostor_path) as f:
            impostor = [sum(n) for n in zip_longest(impostor, map(float, f.read().splitlines()), fillvalue=0)]

    _save_file(r"{0}\bonafide.score".format(output_path), [x / size for x in genuine])
    _save_file(r"{0}\impostor.score".format(output_path), [x / size for x in impostor])


def all_combinations(elements, min_group_size, max_group_size):
    for L in range(min_group_size, max_group_size + 1):
        for subset in itertools.combinations(elements, L):
            yield list(subset)


#single_scores_output_path_template = r'C:\Users\emanu\PycharmProjects\morphClassifier\scores_final_fusion\london\single\{0}\{1}'
diff_output_path_template = r'C:\Users\emanu\PycharmProjects\morphClassifier\scores_paper\morphdb_accomplice\score_fusions\{0}\{1}_diff_subtraction'

#single_scores_template = r'C:\Users\emanu\PycharmProjects\morphClassifier\scores_thesis\{0}\SVC'
diff_scores_template = r'C:\Users\emanu\PycharmProjects\morphClassifier\scores_paper\morphdb_accomplice\{0}_diff_subtraction\SVC'


combinations = list(all_combinations(['LBP224_25', 'LBPH224_25', 'HOG224', 'SIFT224_200', 'SURF224_200'], 2, 5))
for comb in combinations:
    output_path = diff_output_path_template.format(str(len(comb)), '+'.join(comb))
    file_paths = list(map(lambda f: diff_scores_template.format(f), comb))
    score_fusion(file_paths, output_path)
print("Done fusions")
