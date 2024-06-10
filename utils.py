import numpy as np


def calculate_ship2(
        matrix: np.ndarray
) -> tuple[float, np.ndarray]:
    """Calculate SHIP.

    Args:
        matrix (np.ndarray): A matrix of percentages of gene expesssioning
            tumor cells.

    Returns:
        tuple[float, np.ndarray]: SHIP value and a matrix of differences
            between neighboring patches.
    """
    rows, cols = matrix.shape
    N = rows * cols  # すべての patch 数
    diff_matrix = np.zeros((rows, cols))  # 隣接 patch との差の2乗和を格納する行列

    for i in range(rows):
        for j in range(cols):
            a_ij = matrix[i, j]
            neighbors = []

            # 上の patch が存在する場合
            if i > 0:
                neighbors.append(matrix[i - 1, j])
            # 下の patch が存在する場合
            if i < rows - 1:
                neighbors.append(matrix[i + 1, j])
            # 左の patch が存在する場合
            if j > 0:
                neighbors.append(matrix[i, j - 1])
            # 右の patch が存在する場合
            if j < cols - 1:
                neighbors.append(matrix[i, j + 1])

            # 隣接 patch との差の2乗和を計算
            for neighbor in neighbors:
                diff_matrix[i, j] += (a_ij - neighbor) ** 2

    # SHIPを計算
    ship = (1 / (2 * N)) * diff_matrix.sum()
    return ship, diff_matrix


def calculate_ship(matrix):
    rows, cols = matrix.shape
    N = rows * cols  # すべての patch 数
    diff_matrix = np.zeros((rows, cols))  # 隣接 patch との差の2乗和を格納する行列

    for i in range(rows):
        for j in range(cols):
            a_ij = matrix[i, j]
            neighbors = []

            # 上の patch が存在する場合
            if i > 0 and not np.isnan(matrix[i - 1, j]):
                neighbors.append(matrix[i - 1, j])
            # 下の patch が存在する場合
            if i < rows - 1 and not np.isnan(matrix[i + 1, j]):
                neighbors.append(matrix[i + 1, j])
            # 左の patch が存在する場合
            if j > 0 and not np.isnan(matrix[i, j - 1]):
                neighbors.append(matrix[i, j - 1])
            # 右の patch が存在する場合
            if j < cols - 1 and not np.isnan(matrix[i, j + 1]):
                neighbors.append(matrix[i, j + 1])

            # 隣接 patch との差の2乗和を計算
            if neighbors:  # 隣接する有効なピクセルが存在する場合のみ計算
                for neighbor in neighbors:
                    diff_matrix[i, j] += (a_ij - neighbor) ** 2

    # SHIPを計算
    valid_patch_count = np.sum(~np.isnan(matrix))  # 欠損値を含まないパッチ数
    ship = (1 / (2 * valid_patch_count)) * np.nansum(diff_matrix)
    return ship, diff_matrix
