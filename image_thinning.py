def get_p_list(img, i, j):
    p_list = np.zeros(9)
    img_pad = img
    p_list[0] = img_pad[i, j]
    p_list[1] = img_pad[i - 1, j]
    p_list[2] = img_pad[i - 1, j + 1]
    p_list[3] = img_pad[i, j + 1]
    p_list[4] = img_pad[i + 1, j + 1]
    p_list[5] = img_pad[i + 1, j]
    p_list[6] = img_pad[i + 1, j - 1]
    p_list[7] = img_pad[i, j - 1]
    p_list[8] = img_pad[i - 1, j - 1]
    return p_list


def boundary(p_list):
    return np.sum(p_list) - p_list[0]


def count_pattern(p_list):
    """Count 01 patterns in p_list"""
    count = 0
    for i in range(1, np.size(p_list) - 1):
        if p_list[i] == 0 and p_list[i + 1] == 1:
            count += 1
    if p_list[-1] == 0 and p_list[1] == 1:
        count += 1
    return count


def condition1(p_list):
    if not p_list[0]:
        return False
    boundary_points = boundary(p_list)
    return boundary_points >= 2 and boundary_points <= 6 and count_pattern(p_list) == 1 and p_list[1] * p_list[3] * \
                                                                                            p_list[5] == 0 and p_list[
                                                                                                                   3] * \
                                                                                                               p_list[
                                                                                                                   5] * \
                                                                                                               p_list[
                                                                                                                   7] == 0


def condition2(p_list):
    if not p_list[0]:
        return False
    boundary_points = boundary(p_list)
    return boundary_points >= 2 and boundary_points <= 6 and count_pattern(p_list) == 1 and p_list[1] * p_list[3] * \
                                                                                            p_list[7] == 0 and p_list[
                                                                                                                   1] * \
                                                                                                               p_list[
                                                                                                                   5] * \
                                                                                                               p_list[
                                                                                                                   7] == 0


def iteration(img_orig, condition=condition1):
    img_binary = np.uint8(img_orig > 0)
    img_mask = np.ones(img_orig.shape, dtype=np.uint8)
    for i in range(1, img_orig.shape[0] - 1):
        for j in range(1, img_orig.shape[1] - 1):
            p_list = get_p_list(img_binary, i, j)
            if condition(p_list):
                img_mask[i, j] = 0

    return img_mask


def iteration1(img_orig):
    return iteration(img_orig, condition1)


def iteration2(img_orig):
    return iteration(img_orig, condition2)


def remove_mask(img):
    img &= iteration1(img)
    img &= iteration2(img)
    return img


num_pixels_changed = True
while num_pixels_changed:
    sum_orig = np.sum(img1)
    remove_mask(img1)
    num_pixels_changed = np.sum(img1) < sum_orig