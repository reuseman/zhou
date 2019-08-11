from math import floor, log

frequency = 250


def get_pimap(n, cons):
    pi = lambda i: int(-cons / log(n, 2) * i * log(i, 2))
    return [pi(p / n) for p in range(1, n + 1)]


# Here 1 bpm is lost: the part before the first QRS.
def compute_bpm(qrs_annotations_list):
    bpm_list = list()

    for i in range(0, len(qrs_annotations_list) - 1):
        dist = qrs_annotations_list[i] - qrs_annotations_list[i + 1]
        bpm = 60 / (dist / frequency)
        bpm_list.append(bpm)

    return bpm_list


def compute_sy(bpm_list):
    return list(map(lambda x: 63 if x >= 315 else floor(x / 5), bpm_list))


# Here 2 bpm are lost: bpm_0 and bpm_1.
def compute_wv(sy_list):
    wv_list = list()

    for i in range(2, len(sy_list)):
        wy = sy_list[i] + (sy_list[i - 1] * 64) + (sy_list[i - 2] * 4096)
        wv_list.append(wy)

    return wv_list


pi_map = get_pimap(127, 1000000)


def compute_entropy(wv_list):
    nu = list([0] * 127)
    sh2 = list()

    for i in range(0, len(wv_list)):
        nu.pop(0)
        nu.append(wv_list[i])

        # Number of occurrences of an element are counted
        a = dict()
        for j in nu:
            a[j] = (a[j] + 1) if j in a else 1

        k = len(a)
        sh1 = sum([pi_map[a[element] - 1] for element in a])
        sh2.append(k / 127000000 * sh1)

    return sh2


def get_entropy(qrs_annotations_list):
    bpm = compute_bpm(qrs_annotations_list)
    sy = compute_sy(bpm)
    wv = compute_wv(sy)

    return compute_entropy(wv)
