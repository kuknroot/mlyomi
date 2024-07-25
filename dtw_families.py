# %%
# 制作日時
# 2024/07/26
# 参考文献
# [1] https://www.jstage.jst.go.jp/article/pjsai/JSAI2017/0/JSAI2017_2D11/_article/-char/ja/
# [2] https://da.lib.kobe-u.ac.jp/da/kernel/81008373/

# %%
realnumber = int | float
realnumbers = list[realnumber]

# %%
def calc_distance(x: realnumber, y: realnumber, method="abs") -> realnumber:
    
    if method == "abs":
        return abs(x-y)

def calc_dtw(xs: realnumbers, ys: realnumbers, method="abs") -> float:

    large_number: realnumber = 114514
    expand_xs: realnumbers = xs + [large_number]
    expand_ys: realnumbers = ys + [large_number]
    distance_matrix: list[realnumbers] = [[0 for j, y in enumerate(expand_ys)] for i, x in enumerate(expand_xs)]

    i: int
    x: realnumber
    for i, x in enumerate(expand_xs):
        j: int
        y: realnumber
        for j, y in enumerate(expand_ys):
            distance_matrix[i][j] = int(max(i, j)>0)*large_number

    i: int
    x: realnumber
    for i, x in enumerate(expand_xs[1:], 1):
        j: int
        y: realnumber
        for j, y in enumerate(expand_ys[1:], 1):
            distance_term = calc_distance(xs[i-1], ys[j-1], method)
            min_term = min(distance_matrix[i-1][j], distance_matrix[i][j-1], distance_matrix[i-1][j-1])
            distance_matrix[i][j] = distance_term + min_term

    return distance_matrix

def calc_ddtw_list(xs: realnumbers, ys: realnumbers) -> tuple[realnumbers, realnumbers]:

    Dx: realnumbers | list = []
    Dy: realnumbers | list = []

    i: int
    x: realnumber
    for i, x in enumerate(xs[1:-1], 1):
        dx: realnumber = ((x - xs[i-1]) + ((xs[i+1] - xs[i-1]) / 2))/ 2
        Dx.append(dx)

    j: int
    y: realnumber
    for j, y in enumerate(ys[1:-1], 1):
        dy: realnumber = ((y - ys[j-1]) + ((ys[j+1] - ys[j-1]) / 2))/ 2
        Dy.append(dy)

    return Dx, Dy

def calc_idtw_list(xs: realnumbers, ys: realnumbers) -> tuple[realnumbers, realnumbers]:

    Ix: realnumbers | list = [1]
    Iy: realnumbers | list = [1]

    i: int
    x: realnumber
    for i, x in enumerate(xs[1:], 0):
        ix: realnumber = Ix[i] * (x / xs[i])
        Ix.append(ix)

    j: int
    y: realnumber
    for j, y in enumerate(ys[1:], 0):
        iy: realnumber = Iy[i] * (y / ys[j])
        Iy.append(iy)

    return Ix, Iy

def calc_ddtw(xs: realnumbers, ys: realnumbers, method="abs") -> float:

    Dx, Dy = calc_ddtw_list(xs , ys)

    distance_matrix = calc_dtw(Dx , Dy, method)
            
    return distance_matrix

def calc_idtw(xs: realnumbers, ys: realnumbers, method="abs") -> float:

    Ix, Iy = calc_idtw_list(xs , ys)

    distance_matrix = calc_dtw(Ix , Iy, method)
            
    return distance_matrix

if __name__ == '__main__':
    pass
