import numpy as np
from scipy.spatial import ConvexHull
from skimage.color import rgb2lab, rgb2ycbcr
from skimage.draw import polygon


def enforce_label_connectivity(img_lab, labels, K=5):

    dx = [-1, 0, 1, 0]
    dy = [0, -1, 0, 1]

    m_height, m_width, _ = img_lab.shape
    M, N = labels.shape

    SUPSZ = (m_height * m_width) / K

    nlabels = -1 * np.ones((M, N), dtype=np.int32)

    label = 1
    adjlabel = 1

    xvec = np.zeros(m_height * m_width, dtype=np.int32)
    yvec = np.zeros(m_height * m_width, dtype=np.int32)

    m = 0
    n = 0

    for j in range(m_height):
        for k in range(m_width):

            # Find an unlabeled pixel
            if nlabels[m, n] < 0:

                nlabels[m, n] = label
                xvec[0] = k
                yvec[0] = j

                # Find an adjacent existing label (if any)
                for i in range(4):
                    x = xvec[0] + dx[i]
                    y = yvec[0] + dy[i]
                    if 0 <= x < m_width and 0 <= y < m_height:
                        if nlabels[y, x] > 0:
                            adjlabel = nlabels[y, x]

                # Flood fill / region growing
                count = 1
                c = 0
                while c < count:
                    for i in range(4):
                        x = xvec[c] + dx[i]
                        y = yvec[c] + dy[i]
                        if 0 <= x < m_width and 0 <= y < m_height:
                            if (nlabels[y, x] < 0 and
                                labels[m, n] == labels[y, x]):
                                xvec[count] = x
                                yvec[count] = y
                                nlabels[y, x] = label
                                count += 1
                    c += 1

                # Merge small regions
                if count < (SUPSZ / 4):
                    for c in range(count):
                        nlabels[yvec[c], xvec[c]] = adjlabel
                    label -= 1  # cancel this label

                label += 1

            # Advance linear scan
            n += 1
            if n >= m_width:
                n = 0
                m += 1

    return nlabels


def perform_superpixel_slic(img_lab, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, step, compactness):
    n_rows, n_cols, n_channel = img_lab.shape
    numseeds = kseedsl.shape[0]
    img_lab = img_lab.astype(np.float64)

    klabels = np.zeros((n_rows, n_cols), dtype=np.int32)

    clustersize = np.zeros((numseeds, 1), dtype=np.float64)
    inv = np.zeros((numseeds, 1), dtype=np.float64)

    sigmal = np.zeros((numseeds, 1), dtype=np.float64)
    sigmaa = np.zeros((numseeds, 1), dtype=np.float64)
    sigmab = np.zeros((numseeds, 1), dtype=np.float64)
    sigmax = np.zeros((numseeds, 1), dtype=np.float64)
    sigmay = np.zeros((numseeds, 1), dtype=np.float64)
    invwt = 1.0 / ((step / compactness) * (step / compactness))
    distvec = 100000 * np.ones((n_rows, n_cols), np.float64)
    numk = numseeds

    for itr in range(10):
        sigmal.fill(0)
        sigmaa.fill(0)
        sigmab.fill(0)
        sigmax.fill(0)
        sigmay.fill(0)
        clustersize.fill(0)
        inv.fill(0)
        distvec = np.full((n_rows, n_cols), 100000.0)

        for n in range(numk):
            y1 = max(0, int(kseedsy[n, 0] - step))
            y2 = min(n_rows - 1, int(kseedsy[n, 0] + step))
            x1 = max(0, int(kseedsx[n, 0] - step))
            x2 = min(n_cols - 1, int(kseedsx[n, 0] + step))
    
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                dist_lab = (img_lab[y, x, 0] - kseedsl[n, 0]) ** 2 + (img_lab[y, x, 1] - kseedsa[n, 0]) ** 2 + (img_lab[y, x, 2] - kseedsb[n, 0]) ** 2
                dist_xy = (y - kseedsy[n, 0]) ** 2 + (x - kseedsx[n, 0]) ** 2
                dist = dist_lab + dist_xy * invwt

                if dist < distvec[y, x]:
                    distvec[y, x] = dist
                    klabels[y, x] = n

        for r in range(n_rows):
            for c in range(n_cols):
                lbl = klabels[r, c]
                sigmal[lbl, 0] += img_lab[r, c, 0]
                sigmaa[lbl, 0] += img_lab[r, c, 1]
                sigmab[lbl, 0] += img_lab[r, c, 2]
                sigmax[lbl, 0] += c
                sigmay[lbl, 0] += r
                clustersize[lbl, 0] += 1
                        
        for m in range(numseeds):
                if clustersize[m, 0] <= 0:
                    clustersize[m, 0] = 1
                inv[m, 0] = 1.0 / clustersize[m, 0]

        for m in range(numseeds):
            kseedsl[m, 0] = sigmal[m, 0] * inv[m, 0]
            kseedsa[m, 0] = sigmaa[m, 0] * inv[m, 0]
            kseedsb[m, 0] = sigmab[m, 0] * inv[m, 0]
            kseedsx[m, 0] = sigmax[m, 0] * inv[m, 0]
            kseedsy[m, 0] = sigmay[m, 0] * inv[m, 0]

    return klabels


def MI(a, b):

    Ma, Na = a.shape
    Mb, Nb = b.shape
    M = min(Ma, Mb)
    N = min(Na, Nb)

    a = a[:M, :N]
    b = b[:M, :N]

    # Initialize histograms
    hab = np.zeros((256, 256), dtype=np.float64)
    ha = np.zeros(256, dtype=np.float64)
    hb = np.zeros(256, dtype=np.float64)

    # Normalization to [0,1]
    a_min, a_max = a.min(), a.max()
    if a_max != a_min:
        a = (a - a_min) / (a_max - a_min)
    else:
        a = np.zeros((M, N))

    b_min, b_max = b.min(), b.max()
    if b_max != b_min:
        b = (b - b_min) / (b_max - b_min)
    else:
        b = np.zeros((M, N))

    # Quantization
    a = (a * 255).astype(np.int16)
    b = (b * 255).astype(np.int16)

    # Histogram accumulation
    for i in range(M):
        for j in range(N):
            ix = a[i, j]
            iy = b[i, j]
            hab[ix, iy] += 1
            ha[ix] += 1
            hb[iy] += 1

    # Joint entropy H(a,b)
    hsum = hab.sum()
    p = hab / hsum
    nz = p > 0
    Hab = -np.sum(p[nz] * np.log(p[nz]))

    # Entropy H(a)
    hsum = ha.sum()
    p = ha / hsum
    nz = p > 0
    Ha = -np.sum(p[nz] * np.log(p[nz]))

    # Entropy H(b)
    hsum = hb.sum()
    p = hb / hsum
    nz = p > 0
    Hb = -np.sum(p[nz] * np.log(p[nz]))

    # Mutual information
    mi = Ha + Hb - Hab

    # Normalized mutual information
    mi_sum = 2 * mi / (Ha + Hb) if (Ha + Hb) != 0 else 0.0

    return Ha, mi_sum

def min_bound_rect(x, y, metric='a'):

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError("x and y must have the same length")

    n = x.size

    if n == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    if n == 1:
        return np.repeat(x, 5), np.repeat(y, 5), 0.0, 0.0

    if n == 2:
        rectx = x[[0, 1, 1, 0, 0]]
        recty = y[[0, 1, 1, 0, 0]]
        perimeter = 2 * np.hypot(x[1] - x[0], y[1] - y[0])
        return rectx, recty, 0.0, perimeter

    points = np.column_stack((x, y))
    hull = ConvexHull(points)
    hull_pts = points[hull.vertices]

    hull_pts = np.vstack([hull_pts, hull_pts[0]])

    edges = hull_pts[1:] - hull_pts[:-1]
    edge_angles = np.arctan2(edges[:, 1], edges[:, 0])
    edge_angles = np.mod(edge_angles, np.pi / 2.0)
    edge_angles = np.unique(edge_angles)

    def R(theta):
        return np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])

    best_metric = np.inf
    best_area = np.inf
    best_perimeter = np.inf
    best_rect = None

    for theta in edge_angles:
        rot = R(-theta)
        rot_pts = hull_pts @ rot.T

        min_xy = rot_pts.min(axis=0)
        max_xy = rot_pts.max(axis=0)

        width, height = max_xy - min_xy
        area = width * height
        perimeter = 2 * (width + height)

        current_metric = area if metric == 'a' else perimeter

        if current_metric < best_metric:
            best_metric = current_metric
            best_area = area
            best_perimeter = perimeter

            rect = np.array([
                [min_xy[0], min_xy[1]],
                [max_xy[0], min_xy[1]],
                [max_xy[0], max_xy[1]],
                [min_xy[0], max_xy[1]],
                [min_xy[0], min_xy[1]]
            ])

            best_rect = rect @ rot

    rectx = best_rect[:, 0]
    recty = best_rect[:, 1]

    return rectx, recty, best_area, best_perimeter


def calculate_rsei(reference_image, deformed_image):

    K = 5
    compactness = 20

    ref_image = reference_image.astype(np.float64)
    def_image = deformed_image.astype(np.float64)

    n_rows, n_cols, _ = reference_image.shape
    image_size = n_rows * n_cols

    img_lab = rgb2lab(ref_image)

    superpixel_size = image_size / K
    step = np.uint32(np.sqrt(superpixel_size))

    xstrips = np.uint32(n_cols / step)
    ystrips = np.uint32(n_rows / step)
    xstrips_adderr = np.float64(n_cols / xstrips)
    ystrips_adderr = np.float64(n_rows / ystrips)
    numseeds = xstrips * ystrips

    kseedsx = np.zeros((numseeds, 1), dtype=np.float64)
    kseedsy = np.zeros((numseeds, 1), dtype=np.float64)
    kseedsl = np.zeros((numseeds, 1), dtype=np.float64)
    kseedsa = np.zeros((numseeds, 1), dtype=np.float64)
    kseedsb = np.zeros((numseeds, 1), dtype=np.float64)

    n = 0
    for y in range(ystrips):
        for x in range(xstrips):
            kseedsx[n, 0] = (x - 0.5) * xstrips_adderr
            kseedsy[n, 0] = (y - 0.5) * ystrips_adderr

            px = int(np.fix(kseedsx[n, 0]))
            py = int(np.fix(kseedsy[n, 0]))

            kseedsl[n, 0] = img_lab[py, px, 0]
            kseedsa[n, 0] = img_lab[py, px, 1]
            kseedsb[n, 0] = img_lab[py, px, 2]
            n += 1
    n = 0

    klabels = perform_superpixel_slic(img_lab, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, step, compactness)

    nlabels = enforce_label_connectivity(img_lab, klabels, K)

    max_label = int(nlabels.max())
    Ha_l = np.zeros(max_label, dtype=np.float64)
    mi_sum_l = np.zeros(max_label, dtype=np.float64)

    for label in range(1, max_label + 1):
        
        new_nlabels = nlabels.copy()
        new_nlabels[new_nlabels != label] = 0
        new_nlabels1 = (new_nlabels == label).astype(np.uint8)

        img_y1 = ref_image.copy()
        img_y2 = def_image.copy()

        img_y1 = rgb2ycbcr(img_y1).astype(np.float64)
        img_y2 = rgb2ycbcr(img_y2).astype(np.float64)

        new_nlabels2 = new_nlabels1.astype(bool)

        r, c = np.where(new_nlabels2)

        if len(r) == 0:
            continue

        rectx, recty, _, _ = min_bound_rect(c, r)

        rr, cc = polygon(recty, rectx, new_nlabels2.shape)
        new_nlabels2 = np.zeros_like(new_nlabels2, dtype=bool)
        new_nlabels2[rr, cc] = True

        img_y11 = img_y1[:, :, 0] * new_nlabels2
        img_y21 = img_y2[:, :, 0] * new_nlabels2

        img_y11 = img_y11[~np.all(img_y11 == 0, axis=1)]
        img_y21 = img_y21[~np.all(img_y21 == 0, axis=1)]

        img_y11 = img_y11[:, ~np.all(img_y11 == 0, axis=0)]
        img_y21 = img_y21[:, ~np.all(img_y21 == 0, axis=0)]

        Ha, mi_sum = MI(img_y11, img_y21)

        Ha_l[label - 1] = Ha
        mi_sum_l[label - 1] = mi_sum

    Ha_sum = np.sum(Ha_l)

    new_Ha = Ha_l / Ha_sum
    pic_MI = np.sum(new_Ha * mi_sum_l)

    return pic_MI

    

    









