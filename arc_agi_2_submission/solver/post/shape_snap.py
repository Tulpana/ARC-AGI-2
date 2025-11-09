def _neighbors4(H, W, r, c):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and 0 <= cc < W:
            yield rr, cc


def snap_lonely_cells(candidate):
    """
    If a cell's colour is unique among its 4-neighborhood and differs from the
    majority colour of its 2-step neighborhood, snap it to that majority.
    Dimension- and palette-agnostic.
    """

    H, W = len(candidate), len(candidate[0])
    out = [row[:] for row in candidate]
    for r in range(H):
        for c in range(W):
            v = candidate[r][c]
            # count immediate same-colour neighbors
            same = sum(1 for rr, cc in _neighbors4(H, W, r, c) if candidate[rr][cc] == v)
            if same > 0:
                continue
            # 2-step majority vote
            counts = {}
            frontier = [(r, c)]
            seen = {(r, c)}
            for _ in range(2):
                nxt = []
                for rr, cc in frontier:
                    for nr, nc in _neighbors4(H, W, rr, cc):
                        if (nr, nc) in seen:
                            continue
                        seen.add((nr, nc))
                        nxt.append((nr, nc))
                        counts[candidate[nr][nc]] = counts.get(candidate[nr][nc], 0) + 1
                frontier = nxt
            if counts:
                maj = max(counts, key=counts.get)
                if maj != v:
                    out[r][c] = maj
    return out
