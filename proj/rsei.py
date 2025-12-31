import numpy as np
from scipy.spatial import ConvexHull
from skimage.color import rgb2lab, rgb2ycbcr
from skimage.draw import polygon
from skimage.segmentation import slic

def enforce_label_connectivity(img_Lab, labels, K):
    dx = np.array([-1, 0, 1, 0], dtype=np.int32)
    dy = np.array([0, -1, 0, 1], dtype=np.int32)

    h, w, _ = img_Lab.shape
    SUPSZ = (h * w) / K

    nlabels = -np.ones((h, w), dtype=np.int32)

    xvec = np.empty(h * w, dtype=np.int32)
    yvec = np.empty(h * w, dtype=np.int32)

    label = 1
    adjlabel = 1

    m = n = 0

    for j in range(h):
        for k in range(w):

            if nlabels[m, n] < 0:
                nlabels[m, n] = label
                xvec[0] = k
                yvec[0] = j

                for i in range(4):
                    x = k + dx[i]
                    y = j + dy[i]
                    if 0 <= x < w and 0 <= y < h:
                        if nlabels[y, x] > 0:
                            adjlabel = nlabels[y, x]

                count = 1
                c = 0
                while c < count:
                    cx = xvec[c]
                    cy = yvec[c]
                    for i in range(4):
                        x = cx + dx[i]
                        y = cy + dy[i]
                        if 0 <= x < w and 0 <= y < h:
                            if nlabels[y, x] < 0 and labels[m, n] == labels[y, x]:
                                nlabels[y, x] = label
                                xvec[count] = x
                                yvec[count] = y
                                count += 1
                    c += 1

                if count < SUPSZ / 4:
                    for c in range(count):
                        nlabels[yvec[c], xvec[c]] = adjlabel
                    label -= 1

                label += 1

            n += 1
            if n == w:
                n = 0
                m += 1

    return nlabels


def perform_superpixel_slic(img_lab, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, step, compactness):

    H, W, _ = img_lab.shape
    numseeds = kseedsl.shape[0]

    klabels = -np.ones((H, W), dtype=np.int32)
    distvec = np.full((H, W), np.inf)

    invwt = 1.0 / ((step / compactness) ** 2)

    for _ in range(10):

        distvec.fill(np.inf)

        for n in range(numseeds):
            y0, x0 = int(kseedsy[n, 0]), int(kseedsx[n, 0])

            y1 = max(0, y0 - step)
            y2 = min(H, y0 + step)
            x1 = max(0, x0 - step)
            x2 = min(W, x0 + step)

            patch = img_lab[y1:y2, x1:x2]

            dl = patch[..., 0] - kseedsl[n, 0]
            da = patch[..., 1] - kseedsa[n, 0]
            db = patch[..., 2] - kseedsb[n, 0]

            yy, xx = np.mgrid[y1:y2, x1:x2]
            dxy = (yy - y0) ** 2 + (xx - x0) ** 2

            dist = dl**2 + da**2 + db**2 + dxy * invwt

            mask = dist < distvec[y1:y2, x1:x2]
            distvec[y1:y2, x1:x2][mask] = dist[mask]
            klabels[y1:y2, x1:x2][mask] = n

        # Update seeds
        for n in range(numseeds):
            mask = klabels == n
            if not np.any(mask):
                continue
            pts = img_lab[mask]
            ys, xs = np.where(mask)

            kseedsl[n, 0] = pts[:, 0].mean()
            kseedsa[n, 0] = pts[:, 1].mean()
            kseedsb[n, 0] = pts[:, 2].mean()
            kseedsx[n, 0] = xs.mean()
            kseedsy[n, 0] = ys.mean()

    return klabels



def MI(a, b):
    M = min(a.shape[0], b.shape[0])
    N = min(a.shape[1], b.shape[1])
    a = a[:M, :N]
    b = b[:M, :N]

    # Normalize to [0,1]
    def normalize(x):
        xmin, xmax = x.min(), x.max()
        if xmax > xmin:
            return (x - xmin) / (xmax - xmin)
        return np.zeros_like(x)

    a = normalize(a)
    b = normalize(b)

    a = (a * 255).astype(np.uint8)
    b = (b * 255).astype(np.uint8)

    # Joint histogram (vectorized)
    hab, _, _ = np.histogram2d(
        a.ravel(), b.ravel(),
        bins=256,
        range=[[0, 255], [0, 255]]
    )

    ha = hab.sum(axis=1)
    hb = hab.sum(axis=0)

    def entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    hab /= hab.sum()
    ha /= ha.sum()
    hb /= hb.sum()

    Hab = entropy(hab)
    Ha = entropy(ha)
    Hb = entropy(hb)

    mi = Ha + Hb - Hab
    mi_sum = 2 * mi / (Ha + Hb) if (Ha + Hb) > 0 else 0.0

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

    K = 20
    compactness = 20

    ref_image = reference_image.astype(np.float64)
    def_image = deformed_image.astype(np.float64)

    n_rows, n_cols, _ = reference_image.shape
    image_size = n_rows * n_cols

    img_lab = rgb2lab(ref_image)

    superpixel_size = image_size / K
    step = int(np.sqrt(superpixel_size))

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

    # time_start = default_timer()

    klabels = perform_superpixel_slic(img_lab, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, step, compactness)
    # klabels = slic(img_lab, K, compactness, max_size_factor=superpixel_size)

    nlabels = enforce_label_connectivity(img_lab, klabels, K)

    # time_end = default_timer()
    # print(f"\nTime elapsed for processing: {time_end - time_start:.2f} seconds\n")

    max_label = int(nlabels.max())
    Ha_l = np.zeros(max_label, dtype=np.float64)
    mi_sum_l = np.zeros(max_label, dtype=np.float64)

    img_y1 = rgb2ycbcr(ref_image)[:, :, 0]
    img_y2 = rgb2ycbcr(def_image)[:, :, 0]

    for label in range(1, max_label + 1):
        mask = (nlabels == label)
        if not np.any(mask):
            continue

        r, c = np.where(mask)
        rectx, recty, _, _ = min_bound_rect(c, r)

        rr, cc = polygon(recty, rectx, mask.shape)
        rect_mask = np.zeros_like(mask)
        rect_mask[rr, cc] = True

        y1 = img_y1 * rect_mask
        y2 = img_y2 * rect_mask

        y1 = y1[np.any(y1, axis=1)]
        y2 = y2[np.any(y2, axis=1)]

        y1 = y1[:, np.any(y1, axis=0)]
        y2 = y2[:, np.any(y2, axis=0)]

        Ha, mi_sum = MI(y1, y2)

        Ha_l[label - 1] = Ha
        mi_sum_l[label - 1] = mi_sum

    Ha_sum = np.sum(Ha_l)

    new_Ha = Ha_l / Ha_sum
    pic_MI = np.sum(new_Ha * mi_sum_l)

    return pic_MI

    

    









