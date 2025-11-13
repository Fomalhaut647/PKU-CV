from PIL import Image
import numpy as np
import scipy
import matplotlib
import os

# 检测是否有图形界面可用
has_display = os.environ.get("DISPLAY") is not None

# 根据环境选择后端
backend_set = False
if has_display:
    # 如果有图形界面，尝试使用交互式后端
    for backend in ["TkAgg", "Qt5Agg"]:
        try:
            matplotlib.use(backend)
            import matplotlib.pyplot as plt

            plt.ion()  # 启用交互模式
            backend_set = True
            break
        except (ImportError, ValueError):
            continue

# 如果交互式后端都不可用，使用非交互式后端
if not backend_set:
    matplotlib.use("Agg")  # 使用 Agg 后端（非交互式）
    import matplotlib.pyplot as plt

    has_display = False  # 强制设置为无显示模式
    # 注意：在 headless 模式下，plt.show() 不会显示窗口，但可以保存图像

import scipy

# 用于生成唯一文件名的计数器
_visualization_counter = {"matches": 0, "normed": 0, "original": 0}


def show_or_save(fig, filename=None, counter_key=None):
    """
    根据环境自动选择显示或保存图像
    如果有图形界面则显示，否则保存到文件
    """
    if has_display:
        plt.show()
    else:
        # headless 模式下保存图像
        if filename is None:
            # 生成默认文件名（基于时间戳）
            import time

            filename = f"output_{int(time.time())}.png"
        elif counter_key and counter_key in _visualization_counter:
            # 如果有计数器键，添加序号以避免文件名冲突
            _visualization_counter[counter_key] += 1
            name, ext = (
                filename.rsplit(".", 1) if "." in filename else (filename, "png")
            )
            filename = f"{name}_{_visualization_counter[counter_key]}.{ext}"
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"The image has been saved to: {filename}")
        plt.close(fig)  # 关闭图形以释放内存


def visualize_matches(I1, I2, matches):
    # display two images side-by-side with matches
    # this code is to help you visualize the matches, you don't need
    # to use it to produce the results for the assignment

    I3 = np.zeros((I1.size[1], I1.size[0] * 2, 3))
    I3[:, : I1.size[0], :] = I1
    I3[:, I1.size[0] :, :] = I2
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.imshow(np.array(I3).astype(np.uint8))
    ax.plot(matches[:, 0], matches[:, 1], "+r")
    ax.plot(matches[:, 2] + I1.size[0], matches[:, 3], "+r")
    ax.plot(
        [matches[:, 0], matches[:, 2] + I1.size[0]], [matches[:, 1], matches[:, 3]], "r"
    )
    show_or_save(fig, "matches.png", counter_key="matches")


def normalize_points(pts):
    # Normalize points
    # 1. calculate mean and std
    # 2. build a transformation matrix
    # :return normalized_pts: normalized points
    # :return T: transformation matrix from original to normalized points

    mu = np.mean(pts, axis=0)
    scale = 1 / np.sqrt(np.mean((pts - mu) ** 2))

    T = np.array(
        [
            [scale, 0, -scale * mu[0]],
            [0, scale, -scale * mu[1]],
            [0, 0, 1],
        ]
    )

    homo_pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
    normalized_pts = (T @ homo_pts.T).T[:, :2]

    return normalized_pts, T


def fit_fundamental(matches, normed=True):
    # Calculate fundamental matrix from ground truth matches
    # 1. (normalize points if necessary)
    # 2. (x2, y2, 1) * F * (x1, y1, 1)^T = 0 -> AX = 0
    # X = (f_11, f_12, ..., f_33)
    # build A(N x 9) from matches(N x 4) according to Eight-Point Algorithm
    # 3. use SVD (np.linalg.svd) to decomposite the matrix
    # 4. take the smallest eigen vector(9, ) as F(3 x 3)
    # 5. use SVD to decomposite F, set the smallest eigenvalue as 0, and recalculate F
    # 6. Report your fundamental matrix results

    p1, p2 = matches[:, :2], matches[:, 2:]

    if normed:
        normed_p1, T1 = normalize_points(p1)
        normed_p2, T2 = normalize_points(p2)

        x1, y1 = normed_p1[:, 0], normed_p1[:, 1]
        x2, y2 = normed_p2[:, 0], normed_p2[:, 1]

    else:
        x1, y1 = p1[:, 0], p2[:, 1]
        x2, y2 = p2[:, 0], p2[:, 1]
        T1 = T2 = np.eye(3)

    A = np.stack(
        (x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, np.ones_like(x1)),
        axis=1,
    )

    U, S, Vh = np.linalg.svd(A)
    f = Vh[-1]
    F_prime = f.reshape(3, 3)

    U_F, S_F, Vh_F = np.linalg.svd(F_prime)
    S_F[2] = 0
    F_rank2 = U_F @ np.diag(S_F) @ Vh_F
    F = T2.T @ F_rank2 @ T1

    return F


def visualize_fundamental(matches, F, I1, I2, normed=True):
    # Visualize the fundamental matrix in image 2
    N = len(matches)
    M = np.c_[matches[:, 0:2], np.ones((N, 1))].transpose()
    L1 = np.matmul(F, M).transpose()  # transform points from
    # the first image to get epipolar lines in the second image

    # find points on epipolar lines L closest to matches(:,3:4)
    l = np.sqrt(L1[:, 0] ** 2 + L1[:, 1] ** 2)
    L = np.divide(L1, np.kron(np.ones((3, 1)), l).transpose())  # rescale the line
    pt_line_dist = np.multiply(L, np.c_[matches[:, 2:4], np.ones((N, 1))]).sum(axis=1)
    closest_pt = matches[:, 2:4] - np.multiply(
        L[:, 0:2], np.kron(np.ones((2, 1)), pt_line_dist).transpose()
    )

    # find endpoints of segment on epipolar line (for display purposes)
    pt1 = (
        closest_pt - np.c_[L[:, 1], -L[:, 0]] * 10
    )  # offset from the closest point is 10 pixels
    pt2 = closest_pt + np.c_[L[:, 1], -L[:, 0]] * 10

    # display points and segments of corresponding epipolar lines
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.imshow(np.array(I2).astype(np.uint8))
    ax.plot(matches[:, 2], matches[:, 3], "+r")
    ax.plot([matches[:, 2], closest_pt[:, 0]], [matches[:, 3], closest_pt[:, 1]], "r")
    ax.plot([pt1[:, 0], pt2[:, 0]], [pt1[:, 1], pt2[:, 1]], "g")
    if normed:
        show_or_save(fig, "fundamental_normed.png", counter_key="normed")
    else:
        show_or_save(fig, "fundamental_original.png", counter_key="original")


def evaluate_fundamental(matches, F):
    N = len(matches)
    points1, points2 = matches[:, :2], matches[:, 2:]
    points1_homogeneous = np.concatenate([points1, np.ones((N, 1))], axis=1)
    points2_homogeneous = np.concatenate([points2, np.ones((N, 1))], axis=1)
    product = np.dot(np.dot(points2_homogeneous, F), points1_homogeneous.T)
    diag = np.diag(product)
    residual = np.mean(diag**2)
    print(residual)
    return residual


## Task 0: Load data and visualize
print("-" * 20, "Task 0 start", "-" * 20)
## load images and match files for the first example
## matches[:, :2] is a point in the first image
## matches[:, 2:] is a corresponding point in the second image

library_image1 = Image.open("data/library1.jpg")
library_image2 = Image.open("data/library2.jpg")
library_matches = np.loadtxt("data/library_matches.txt")

lab_image1 = Image.open("data/lab1.jpg")
lab_image2 = Image.open("data/lab2.jpg")
lab_matches = np.loadtxt("data/lab_matches.txt")

## Visualize matches
visualize_matches(library_image1, library_image2, library_matches)
visualize_matches(lab_image1, lab_image2, lab_matches)

print("-" * 20, "Task 0 end", "-" * 20, end="\n\n")


## Task 1: Fundamental matrix
print("-" * 20, "Task 1 start", "-" * 20)
## display second image with epipolar lines reprojected from the first image

# first, fit fundamental matrix to the matches
# Report your fundamental matrices, visualization and evaluation results
library_F_normed = fit_fundamental(
    library_matches, normed=True
)  # this is a function that you should write
visualize_fundamental(
    library_matches, library_F_normed, library_image1, library_image2, normed=True
)
assert evaluate_fundamental(library_matches, library_F_normed) < 0.5

lab_F_normed = fit_fundamental(
    lab_matches, normed=True
)  # this is a function that you should write
visualize_fundamental(lab_matches, lab_F_normed, lab_image1, lab_image2, normed=True)
assert evaluate_fundamental(lab_matches, lab_F_normed) < 0.5

library_F = fit_fundamental(
    library_matches, normed=False
)  # this is a function that you should write
visualize_fundamental(
    library_matches, library_F, library_image1, library_image2, normed=False
)
evaluate_fundamental(library_matches, library_F)

lab_F = fit_fundamental(
    lab_matches, normed=False
)  # this is a function that you should write
visualize_fundamental(lab_matches, lab_F, lab_image1, lab_image2, normed=False)
evaluate_fundamental(lab_matches, lab_F)

print("-" * 20, "Task 1 end", "-" * 20, end="\n\n")


## Task 2: Camera Calibration
print("-" * 20, "Task 2 start", "-" * 20)


def calc_projection(points_2d, points_3d):
    # Calculate camera projection matrices
    # 1. Points_2d = P * Points_3d -> AX = 0
    # X = (p_11, p_12, ..., p_34) is flatten of P
    # build matrix A(2*N, 12) from points_2d
    # 2. SVD decomposite A
    # 3. take the eigen vector(12, ) of smallest eigen value
    # 4. return projection matrix(3, 4)
    # :param points_2d: 2D points N x 2
    # :param points_3d: 3D points N x 3
    # :return P: projection matrix

    N = points_2d.shape[0]
    A = np.zeros((2 * N, 12))

    x = points_2d[:, 0]
    y = points_2d[:, 1]
    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]

    A[::2, 0] = X
    A[::2, 1] = Y
    A[::2, 2] = Z
    A[::2, 3] = 1
    A[::2, 8] = -x * X
    A[::2, 9] = -x * Y
    A[::2, 10] = -x * Z
    A[::2, 11] = -x

    A[1::2, 4] = X
    A[1::2, 5] = Y
    A[1::2, 6] = Z
    A[1::2, 7] = 1
    A[1::2, 8] = -y * X
    A[1::2, 9] = -y * Y
    A[1::2, 10] = -y * Z
    A[1::2, 11] = -y

    U, S, Vh = np.linalg.svd(A)
    p = Vh[-1]
    P = p.reshape(3, 4)

    return P


def rq_decomposition(P):
    # Use RQ decomposition to calculte K, R, T
    # 1. perform RQ decomposition on left-most 3x3 matrix of P(3 x 4) to get K, R
    # 2. calculate T by P = K[R|T]
    # 3. normalize to set K[2, 2] = 1
    # :param P: projection matrix
    # :return K, R, T: camera matrices

    M = P[:, :-1]
    K, R = scipy.linalg.rq(M)
    T = np.linalg.inv(K) @ P[:, 3]

    # 在 rq_decomposition() 函数中，归一化时，为了确保旋转矩阵 R 的行列式仍然是 1 且 K[R|T] = alpha * P 仍然成立，
    # 先执行 K /= scale 完成归一化第一步
    scale = K[2, 2]
    K /= scale
    # 然后把 K 的前 2 个对角元调整为正数，因为这 2 个元素代表焦距，必须是正数
    S = np.diag(np.sign(np.diag(K)))
    K = K @ S
    R = R @ S
    T = T @ S

    return K, R, T


def evaluate_points(P, points_2d, points_3d):
    # Visualize the actual 2D points and the projected 2D points calculated from
    # the projection matrix
    # You do not need to modify anything in this function, although you can if you
    # want to
    # :param P: projection matrix 3 x 4
    # :param points_2d: 2D points N x 2
    # :param points_3d: 3D points N x 3
    # :return points_3d_proj: project 3D points to 2D by P
    # :return residual: residual of points_3d_proj and points_2d

    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(P, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u - points_2d[:, 0], v - points_2d[:, 1]))
    points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    return points_3d_proj, residual


def triangulate_points(P1, P2, point1, point2):
    # Use linear least squares to triangulation 3d points
    # 1. Solve: point1 = P1 * point_3d
    #           point2 = P2 * point_3d
    # 2. use SVD decomposition to solve linear equations
    # :param P1, P2 (3 x 4): projection matrix of two cameras
    # :param point1, point2: points in two images
    # :return point_3d: 3D points calculated by triangulation

    u1, v1 = point1[0], point1[1]
    u2, v2 = point2[0], point2[1]

    p11, p12, p13 = P1[0], P1[1], P1[2]
    p21, p22, p23 = P2[0], P2[1], P2[2]

    A = np.vstack(
        [
            u1 * p13 - p11,
            v1 * p13 - p12,
            u2 * p23 - p21,
            v2 * p23 - p22,
        ]
    )

    U, S, Vh = np.linalg.svd(A)
    X_homo = Vh[-1]
    point_3d = X_homo[:3] / X_homo[3]

    return point_3d


lab_points_3d = np.loadtxt("data/lab_3d.txt")

projection_matrix = dict()
for key, points_2d in zip(["lab_a", "lab_b"], [lab_matches[:, :2], lab_matches[:, 2:]]):
    P = calc_projection(points_2d, lab_points_3d)
    points_3d_proj, residual = evaluate_points(P, points_2d, lab_points_3d)
    distance = np.mean(np.linalg.norm(points_2d - points_3d_proj))
    # Check: residual should be < 20 and distance should be < 4
    assert residual < 20.0 and distance < 4.0
    print()
    print(f"residual: {residual}")
    print(f"distance: {distance}")
    projection_matrix[key] = P
print("Task 2 pass")

print("-" * 20, "Task 2 end", "-" * 20, end="\n\n")

## Task 3
print("-" * 20, "Task 3 start", "-" * 20)
## Camera Centers
projection_library_a = np.loadtxt("data/library1_camera.txt")
projection_library_b = np.loadtxt("data/library2_camera.txt")
projection_matrix["library_a"] = projection_library_a
projection_matrix["library_b"] = projection_library_b

for i, P in enumerate(projection_matrix.values()):
    # Paste your K, R, T results in your report
    K, R, T = rq_decomposition(P)
    print(f"K{i+1}: {K}", f"R{i+1}: {R}", f"T{i+1}: {T}", sep="\n", end="\n\n")

print("-" * 20, "Task 3 end", "-" * 20, end="\n\n")


## Task 4: Triangulation
print("-" * 20, "Task 4 start", "-" * 20)
lab_points_3d_estimated = []
residual_3ds = []
for point_2d_a, point_2d_b, point_3d_gt in zip(
    lab_matches[:, :2], lab_matches[:, 2:], lab_points_3d
):
    point_3d_estimated = triangulate_points(
        projection_matrix["lab_a"], projection_matrix["lab_b"], point_2d_a, point_2d_b
    )

    # Residual between ground truth and estimated 3D points
    residual_3d = np.sum(np.linalg.norm(point_3d_gt - point_3d_estimated))
    assert residual_3d < 0.1
    print(f"residual_3d: {residual_3d}")
    residual_3ds.append(residual_3d)
    lab_points_3d_estimated.append(point_3d_estimated)
residual_3ds = np.array(residual_3ds)
mean_residual_3d = np.mean(residual_3ds)
max_residual_3d = np.max(residual_3ds)
print(f"mean of residual_3d: {mean_residual_3d}")
print(f"max of residual_3d: {max_residual_3d}")

# Residual between re-projected and observed 2D points
lab_points_3d_estimated = np.stack(lab_points_3d_estimated)
_, residual_a = evaluate_points(
    projection_matrix["lab_a"], lab_matches[:, :2], lab_points_3d_estimated
)
_, residual_b = evaluate_points(
    projection_matrix["lab_b"], lab_matches[:, 2:], lab_points_3d_estimated
)
assert residual_a < 20 and residual_b < 20
print(f"residual_a: {residual_a}")
print(f"residual_b: {residual_b}")

library_points_3d_estimated = []
for point_2d_a, point_2d_b in zip(library_matches[:, :2], library_matches[:, 2:]):
    point_3d_estimated = triangulate_points(
        projection_matrix["library_a"],
        projection_matrix["library_b"],
        point_2d_a,
        point_2d_b,
    )
    library_points_3d_estimated.append(point_3d_estimated)

# Residual between re-projected and observed 2D points
library_points_3d_estimated = np.stack(library_points_3d_estimated)
_, residual_a = evaluate_points(
    projection_matrix["library_a"], library_matches[:, :2], library_points_3d_estimated
)
_, residual_b = evaluate_points(
    projection_matrix["library_b"], library_matches[:, 2:], library_points_3d_estimated
)
assert residual_a < 30 and residual_b < 30
print(f"residual_a: {residual_a}")
print(f"residual_b: {residual_b}")
print("Task 4 pass")

print("-" * 20, "Task 4 end", "-" * 20, end="\n\n")


## Task 5: Fundamental matrix estimation without ground-truth matches
print("-" * 20, "Task 5 start", "-" * 20)
import cv2


def fit_fundamental_without_gt(image1, image2):
    # Calculate fundamental matrix without groundtruth matches
    # 1. convert the images to gray
    # 2. compute SIFT keypoints and descriptors
    # 3. match descriptors with Brute Force Matcher
    # 4. select good matches
    # 5. extract matched keypoints
    # 6. compute fundamental matrix with RANSAC
    # :param image1, image2: two-view images
    # :return fundamental_matrix
    # :return matches: selected matched keypoints

    # 1. convert the images to gray
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    # 2. compute SIFT keypoints and descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 3. match descriptors with Brute Force Matcher
    # k=2 表示为每个 des1 中的描述子找到两个最佳匹配
    bf = cv2.BFMatcher()
    matches_knn = bf.knnMatch(des1, des2, k=2)

    # 4. select good matches
    good_matches = []
    for m, n in matches_knn:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 5. extract matched keypoints
    # 至少需要 8 个点才能计算基础矩阵
    if len(good_matches) < 8:
        print(f"Not enough good matches found - {len(good_matches)}/8")
        return None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    # 6. compute fundamental matrix with RANSAC
    # cv2.FM_RANSAC 表示使用 RANSAC 算法
    # 3.0 是 RANSAC 阈值 (in pixels)
    # 0.99 是置信度
    fundamental_matrix, mask = cv2.findFundamentalMat(
        pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99
    )

    if fundamental_matrix is None:
        print("RANSAC failed to find fundamental matrix.")
        return None, None

    # 使用 mask 筛选出 RANSAC 认定的局内点 (inliers)
    inlier_pts1 = pts1[mask.ravel() == 1]
    inlier_pts2 = pts2[mask.ravel() == 1]

    # 将局内点组合成 (N x 4) 的格式，以便 visualize_fundamental 函数使用
    matches_inliers = np.hstack((inlier_pts1, inlier_pts2))

    # 报告结果
    print(f"\n--- RANSAC Results ---")
    print(f"Total SIFT matches found: {len(good_matches)}")
    print(f"Number of inliers: {len(matches_inliers)}")

    # 计算局内点的平均残差 (使用 Task 1 中定义的函数)
    avg_residual = evaluate_fundamental(matches_inliers, fundamental_matrix)
    print(f"Average residual for inliers: {avg_residual}")
    print(f"--- End RANSAC Results ---")

    return fundamental_matrix, matches_inliers


house_image1 = Image.open("data/house1.jpg")
house_image2 = Image.open("data/house2.jpg")

house_F, house_matches = fit_fundamental_without_gt(
    np.array(house_image1), np.array(house_image2)
)
visualize_fundamental(house_matches, house_F, house_image1, house_image2)

guadi_image1 = Image.open("data/gaudi1.jpg")
guadi_image2 = Image.open("data/gaudi2.jpg")

guadi_F, guadi_matches = fit_fundamental_without_gt(
    np.array(guadi_image1), np.array(guadi_image2)
)
visualize_fundamental(guadi_matches, guadi_F, guadi_image1, guadi_image2)

print("-" * 20, "Task 5 end", "-" * 20, end="\n\n")
