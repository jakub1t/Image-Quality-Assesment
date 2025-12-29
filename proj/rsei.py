import numpy as np
from skimage.segmentation import slic
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

    clustersize = np.zeros((numseeds, 1))
    inv = np.zeros((numseeds, 1))

    sigmal = np.zeros((numseeds, 1))
    sigmaa = np.zeros((numseeds, 1))
    sigmab = np.zeros((numseeds, 1))
    sigmax = np.zeros((numseeds, 1))
    sigmay = np.zeros((numseeds, 1))
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

    hab = np.zeros((256, 256), dtype=np.float64)
    ha = np.zeros(256, dtype=np.float64)
    hb = np.zeros(256, dtype=np.float64)

    if np.max(a) != np.min(a):
        a = (a - np.min(a)) / (np.max(a) - np.min(a))
    else:
        a = np.zeros((M, N), dtype=np.float64)

    if np.max(b) != np.min(b):
        b = (b - np.min(b)) / (np.max(b) - np.min(b))
    else:
        b = np.zeros((M, N), dtype=np.float64)

    a = np.int16(a * 255).astype(np.float64) + 1
    b = np.int16(b * 255).astype(np.float64) + 1

    for i in range(M):
        for j in range(N):
            ix = int(a[i, j]) - 1
            iy = int(b[i, j]) - 1
            hab[ix, iy] += 1
            ha[ix] += 1
            hb[iy] += 1

    hsum = np.sum(hab)
    p = hab / hsum
    nz = p > 0
    Hab = -np.sum(p[nz] * np.log(p[nz]))

    hsum = np.sum(ha)
    p = ha / hsum
    nz = p > 0
    Ha = -np.sum(p[nz] * np.log(p[nz]))

    hsum = np.sum(hb)
    p = hb / hsum
    nz = p > 0
    Hb = -np.sum(p[nz] * np.log(p[nz]))

    mi = Ha + Hb - Hab

    mi_sum = 2 * mi / (Ha + Hb)

    return Ha, mi_sum


def minboundrect(x, y, metric='a'):

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    n = len(x)

    if n > 3:
        hull = ConvexHull(np.column_stack((x, y)))
        edges = hull.vertices
        x = x[edges]
        y = y[edges]
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        nedges = len(x) - 1
    elif n > 1:
        nedges = n
        x = np.append(x, x[0])
        y = np.append(y, y[0])
    else:
        nedges = n

    if nedges == 0:
        return np.array([]), np.array([]), [], []

    if nedges == 1:
        rectx = np.kron(np.ones((1, 5), dtype=np.float64), x) #np.repeat(x[0], 5)
        recty = np.kron(np.ones((1, 5), dtype=np.float64), y) #np.repeat(y[0], 5)
        area = 0
        return rectx, recty, 0.0, 0.0

    if nedges == 2:
        rectx = x[[0, 1, 1, 0, 0]]
        recty = y[[0, 1, 1, 0, 0]]
        area = 0
        perimeter = 2 * np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
        return rectx, recty, 0.0, perimeter
    
    def Rmat(theta):
        return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    edgeangles = np.arctan2(dy, dx)
    edgeangles = np.mod(edgeangles, np.pi / 2)
    edgeangles = np.unique(edgeangles)

    xy = np.column_stack((x, y))

    area = np.inf
    perimeter = np.inf
    met = np.inf

    for theta in edgeangles:
        rot = Rmat(-theta)
        xyr = xy @ rot

        xymin = np.min(xyr, axis=0)
        xymax = np.max(xyr, axis=0)

        A_i = np.prod(xymax - xymin)
        P_i = 2 * np.sum(xymax - xymin)

        M_i = A_i if metric == 'a' else P_i

        if M_i < met:
            met = M_i
            area = A_i
            perimeter = P_i

            rect = np.array([xymin, [xymax[0], xymin[1]], xymax, [xymin[0], xymax[1]], xymin])

            rect = rect @ rot.conj().transpose()
            rectx = rect[:, 0]
            recty = rect[:, 1]

    return rectx, recty, area, perimeter


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

            px = int(np.clip(kseedsx[n, 0], 0, n_rows - 1))
            py = int(np.clip(kseedsy[n, 0], 0, n_cols - 1))

            kseedsl[n, 0] = img_lab[py, px, 0]
            kseedsa[n, 0] = img_lab[py, px, 1]
            kseedsb[n, 0] = img_lab[py, px, 2]
            n += 1
    n = 0

    klabels = perform_superpixel_slic(img_lab, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, step, compactness)

    nlabels = enforce_label_connectivity(img_lab, klabels, K)

    img_y1 = rgb2ycbcr(ref_image).astype(np.float64)[:, :, 0]
    img_y2 = rgb2ycbcr(def_image).astype(np.float64)[:, :, 0]

    max_label = int(nlabels.max())
    Ha_l = np.zeros(max_label)
    mi_sum_l = np.zeros(max_label)

    for label in range(1, max_label + 1):
        mask = (nlabels == label)

        if not np.any(mask):
            continue

        r, c = np.where(mask)

        rectx, recty, _, _ = minboundrect(c, r, 'a')

        rr, cc = polygon(recty, rectx, mask.shape)
        rect_mask = np.zeros_like(mask, dtype=bool)
        rect_mask[rr, cc] = True

        img_y11 = img_y1 * rect_mask
        img_y21 = img_y2 * rect_mask

        img_y11 = img_y11[~np.all(img_y11 == 0, axis=1)]
        img_y21 = img_y21[~np.all(img_y21 == 0, axis=1)]

        img_y11 = img_y11[:, ~np.all(img_y11 == 0, axis=0)]
        img_y21 = img_y21[:, ~np.all(img_y21 == 0, axis=0)]

        if img_y11.size == 0 or img_y21.size == 0:
            continue

        Ha, mi_sum = MI(img_y11, img_y21)

        Ha_l[label - 1] = Ha
        mi_sum_l[label - 1] = mi_sum

    Ha_sum = np.sum(Ha_l)
    if Ha_sum == 0:
        return 0.0

    new_Ha = Ha_l / Ha_sum
    pic_MI = np.sum(new_Ha * mi_sum_l)

    return pic_MI

    

    









