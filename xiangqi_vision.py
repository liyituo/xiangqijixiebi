"""用棋子轮廓、占位变化和外观一致性识别象棋着法。"""

import math
import time

import cv2
import numpy as np

from xiangqi_engine import coord_to_square


def physical_square(square, rotated):
    if not rotated:
        return square
    return coord_to_square(8 - (ord(square[0]) - ord('a')), 9 - int(square[1]))


class MoveConfirmation:
    """同一候选着法需在不同采样帧上保持稳定。"""

    def __init__(self):
        self.required_frames = 3
        self.min_duration = 0.35
        self.reset()

    def reset(self):
        self.move = None
        self.count = 0
        self.started_at = 0.0
        self.last_seen_at = 0.0
        self._last_frame_key = None

    def observe(self, move, frame_key, now=None):
        now = time.monotonic() if now is None else now
        if move is None:
            self.reset()
            return False
        if move != self.move or now - self.last_seen_at > 1.5:
            self.reset()
            self.move = move
            self.started_at = now
        self.last_seen_at = now
        if frame_key != self._last_frame_key:
            self.count += 1
            self._last_frame_key = frame_key
        return self.count >= self.required_frames and now - self.started_at >= self.min_duration


class XiangqiMoveDetector:
    """已知棋局下的局部视觉跟踪器，输出相机方向的源格和目标格。"""

    CELL = 64
    PAD = 48
    SIZE = (608, 672)

    def __init__(self):
        self.reset()

    def reset(self):
        self._base_image = None
        self._cache_key = None
        self._baseline = None

    @staticmethod
    def _circle_patch(image, circle):
        x, y, radius = circle
        diameter = max(8, int(round(2 * radius)))
        patch = cv2.getRectSubPix(image, (diameter, diameter), (float(x), float(y)))
        return cv2.resize(patch, (48, 48), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _find_circles(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.0)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1.1, 35,
            param1=80, param2=24, minRadius=14, maxRadius=29,
        )
        if circles is None:
            return []
        return [tuple(float(value) for value in circle) for circle in circles[0]]

    @classmethod
    def _assign_circles(cls, circles, positions, tolerance):
        pairs = []
        for index, circle in enumerate(circles):
            for square, position in positions.items():
                distance = math.hypot(circle[0] - position[0], circle[1] - position[1])
                if distance <= tolerance * cls.CELL:
                    pairs.append((distance, square, index))
        assigned = {}
        used = set()
        for _, square, index in sorted(pairs):
            if square not in assigned and index not in used:
                assigned[square] = circles[index]
                used.add(index)
        return assigned

    @classmethod
    def _nominal_positions(cls):
        return {
            coord_to_square(column, row): (cls.PAD + column * cls.CELL, cls.PAD + (9 - row) * cls.CELL)
            for row in range(10) for column in range(9)
        }

    @classmethod
    def _refine_positions(cls, positions, observed):
        """用基准图中的棋子中心补偿格点的小偏移，仅供图像跟踪使用。"""
        if len(observed) < 6:
            return dict(positions)
        squares = list(observed)
        expected = np.float32([positions[square] for square in squares])
        measured = np.float32([observed[square][:2] for square in squares])
        transform, inliers = cv2.estimateAffine2D(expected, measured, ransacReprojThreshold=cls.CELL * 0.25)
        if transform is None or inliers is None or int(inliers.sum()) < max(6, len(squares) * 0.6):
            return dict(positions)
        all_points = np.float32(list(positions.values()))
        transformed = cv2.transform(all_points.reshape(1, -1, 2), transform)[0]
        if np.max(np.linalg.norm(transformed - all_points, axis=1)) > cls.CELL * 0.85:
            return dict(positions)
        if not 0.65 <= np.linalg.det(transform[:, :2]) <= 1.4:
            return dict(positions)
        return {square: tuple(point) for square, point in zip(positions, transformed)}

    def prepare_base(self, image, board, grid_mapper, rotated=False, ignored_squares=()):
        corners = [grid_mapper.get_square_norm_pos(square) for square in ('a0', 'i0', 'i9', 'a9')]
        key = (image.shape, tuple(tuple(point) for point in corners), board.get_fen(), rotated, tuple(ignored_squares))
        if image is self._base_image and self._cache_key == key:
            return dict(self._baseline['report'])
        height, width = image.shape[:2]
        source = np.float32([[x * width / 1000.0, y * height / 1000.0] for x, y in corners])
        positions = self._nominal_positions()
        destination = np.float32([positions[square] for square in ('a0', 'i0', 'i9', 'a9')])
        transform = cv2.getPerspectiveTransform(source, destination)
        rectified = cv2.warpPerspective(image, transform, self.SIZE)
        pieces = {
            physical_square(coord_to_square(column, row), rotated): piece
            for row, rank in enumerate(board.board)
            for column, piece in enumerate(rank) if piece != '.'
        }
        circles = self._find_circles(rectified)
        observed = self._assign_circles(circles, {square: positions[square] for square in pieces}, 0.75)
        positions = self._refine_positions(positions, observed)
        observed = self._assign_circles(circles, {square: positions[square] for square in pieces}, 0.6)
        radius = float(np.median([circle[2] for circle in observed.values()])) if observed else 21.0
        recovered_red = []
        for square, piece in pieces.items():
            if piece.isupper() and square not in observed:
                circle = (*positions[square], radius)
                if self._red_fraction(rectified, circle) >= 0.02:
                    observed[square] = circle
                    recovered_red.append(square)
        # 保留基准棋子实际中心，使轻微偏离交叉点的静止棋子仍有稳定的比较区域。
        for square, circle in observed.items():
            positions[square] = circle[:2]
        mask = np.zeros(rectified.shape[:2], np.uint8)
        mask[24:-24, 24:-24] = 255
        for x, y, radius in observed.values():
            cv2.circle(mask, (round(x), round(y)), round(radius * 1.4), 0, -1)
        for square in ignored_squares:
            x, y = positions[square]
            cv2.circle(mask, (round(x), round(y)), round(self.CELL * 0.6), 0, -1)
        missing = sorted(set(pieces) - set(observed))
        missing_red = [square for square in missing if pieces[square].isupper()]
        tracked = set(pieces) - set(ignored_squares)
        required = max(1, math.ceil(len(tracked) * 0.85))
        ready = len(set(observed) & tracked) >= required and not missing_red
        report = {
            'ready': ready,
            'visible_pieces': len(observed),
            'expected_pieces': len(pieces),
            'missing_squares': missing,
            'missing_red_squares': missing_red,
            'recovered_red_squares': recovered_red,
            'reason': '基准图就绪' if ready else f'基准图棋子可见度不足 ({len(observed)}/{len(pieces)})，请移开遮挡并核对格点',
        }
        self._base_image = image
        self._cache_key = key
        self._baseline = {
            'image': rectified, 'transform': transform, 'positions': positions,
            'pieces': pieces, 'circles': observed, 'mask': mask, 'report': report,
            'detected_circles': circles,
        }
        return dict(report)

    def detect_from_position(self, image, board, grid_mapper, rotated=False):
        """依据当前棋局的占位和阵营变化恢复已完成的红方着法。"""
        report = self.prepare_base(image, board, grid_mapper, rotated)
        baseline = self._baseline
        current = baseline['image']
        positions = baseline['positions']
        pieces = baseline['pieces']
        circles = self._assign_circles(baseline['detected_circles'], positions, 0.52)
        for square in report['recovered_red_squares']:
            circles.setdefault(square, baseline['circles'][square])
        expected_red = {square for square, piece in pieces.items() if piece.isupper()}
        visible_red = {square for square, circle in circles.items()
                       if self._red_fraction(current, circle) >= 0.005}
        missing_red = expected_red - visible_red
        report['vacated_squares'] = sorted(missing_red)
        if len(missing_red) != 1:
            report['reason'] = '等待单个红子源格清空与新落点同时出现'
            return None, report
        source = next(iter(missing_red))
        source_circle = (*positions[source], 21.0)
        current_lab = cv2.cvtColor(current, cv2.COLOR_BGR2LAB).astype(np.float32)
        if source in circles or not self._is_empty_source(current_lab, current_lab, source_circle):
            report['reason'] = f'源格 {source} 仍有遮挡，等待棋盘背景显露'
            return None, report
        arrivals = visible_red - expected_red
        if len(arrivals) != 1:
            report['reason'] = f'源格 {source} 已清空，等待唯一落点的阵营确认'
            return None, report
        target = next(iter(arrivals))
        x, y, _ = circles[target]
        if math.hypot(x - positions[target][0], y - positions[target][1]) > self.CELL * 0.48:
            report['reason'] = f'目标格 {target} 的棋子尚未落稳'
            return None, report
        captured = pieces.get(target)
        remaining = len(pieces) - int(captured is not None)
        if len(circles) < max(1, math.ceil(remaining * 0.85)):
            report['reason'] = '棋盘可见度不足，等待移开遮挡后核对吃子局面'
            return None, report
        standard_move = physical_square(source, rotated) + physical_square(target, rotated)
        if not board.is_legal_move(standard_move):
            report.update(
                reason=f'走子错误 {source}{target}，请恢复上一回合棋局',
                illegal_move=source + target, illegal_squares=[source, target],
            )
            return None, report
        report.update(
            reason=f'当前棋局占位与合法着法 {source}{target} 一致，剩余棋子 {remaining} 枚',
            best_move=source + target, piece=pieces[source], captured=captured,
            piece_count_after_move=remaining, position_match=True,
        )
        return (source, target), report

    def matches_snapshot(self, reference_image, image, board, grid_mapper, rotated=False, affected_squares=()):
        """核对错误走子后，棋盘是否已恢复到该回合的棋局与画面快照。"""
        self.prepare_base(reference_image if reference_image is not None else image, board, grid_mapper, rotated)
        baseline = self._baseline
        current = cv2.warpPerspective(image, baseline['transform'], self.SIZE)
        if reference_image is not None:
            try:
                current, _ = self._align(current)
            except (cv2.error, ValueError):
                return False, {'reason': '等待移开遮挡并恢复棋盘位置'}
        circles = self._assign_circles(self._find_circles(current), baseline['positions'], 0.52)
        for square, piece in baseline['pieces'].items():
            reference = baseline['circles'].get(square, (*baseline['positions'][square], 21.0))
            circle = circles.get(square, reference)
            red = self._red_fraction(current, circle)
            if piece.isupper() != (red >= 0.005):
                return False, {'reason': f'{square} 的棋子阵营尚未恢复'}
            if square not in circles and not (piece.isupper() and red >= 0.02):
                if reference_image is None:
                    if square in affected_squares:
                        return False, {'reason': f'{square} 的棋子尚未放回'}
                elif self._appearance_similarity(baseline['image'], current, reference, circle) < 0.65:
                    return False, {'reason': f'{square} 的棋子尚未放回'}
            if reference_image is not None:
                before = self._circle_patch(baseline['image'], reference).astype(np.float32)
                after = self._circle_patch(current, circle).astype(np.float32)
                if float(np.mean(np.abs(before - after))) >= 4.0:
                    if self._appearance_similarity(baseline['image'], current, reference, circle) < 0.5:
                        return False, {'reason': f'{square} 的棋子尚未恢复到快照'}
        for square in set(circles) - set(baseline['pieces']):
            if square in affected_squares or self._red_fraction(current, circles[square]) >= 0.005:
                return False, {'reason': f'请移走 {square} 上错误放置的棋子'}
        if len(circles) < max(1, math.ceil(len(baseline['pieces']) * 0.85)):
            return False, {'reason': '等待棋盘完整可见后确认恢复'}
        return True, {'reason': '棋盘已恢复到上一回合快照'}

    def _align(self, current):
        baseline = self._baseline
        gray_base = cv2.cvtColor(baseline['image'], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        transform = np.eye(2, 3, dtype=np.float32)
        quality, transform = cv2.findTransformECC(
            gray_base, gray_current, transform, cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.0001), baseline['mask'], 5,
        )
        if not math.isfinite(quality) or not np.all(np.isfinite(transform)):
            raise ValueError('画面配准未得到有效结果')
        corners = np.float32([[[24, 24], [584, 24], [584, 648], [24, 648]]])
        displacement = float(np.max(np.linalg.norm(cv2.transform(corners, transform) - corners, axis=2)))
        if quality < 0.78 or displacement > self.CELL * 0.2:
            raise ValueError('棋盘画面存在明显遮挡或视角移动，请移开遮挡并核对棋盘定位')
        aligned = cv2.warpAffine(
            current, transform, self.SIZE,
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT_101,
        )
        # 在棋子区域之外估计色彩偏移，补偿整体亮度和轻微白平衡变化。
        stable = baseline['mask'] != 0
        offset = np.median(baseline['image'][stable].astype(np.float32) - aligned[stable], axis=0)
        aligned = np.clip(aligned.astype(np.float32) + offset, 0, 255).astype(np.uint8)
        return aligned, {'alignment_quality': round(float(quality), 3), 'alignment_shift_px': round(displacement, 2)}

    @staticmethod
    def _disk_values(image, circle, inner=0.0, outer=1.0):
        x, y, radius = circle
        radius *= outer
        left = max(0, int(x - radius - 1))
        right = min(image.shape[1], int(x + radius + 2))
        top = max(0, int(y - radius - 1))
        bottom = min(image.shape[0], int(y + radius + 2))
        yy, xx = np.ogrid[top:bottom, left:right]
        distance = (xx - x) ** 2 + (yy - y) ** 2
        selected = (distance <= radius ** 2) & (distance >= (circle[2] * inner) ** 2)
        return image[top:bottom, left:right][selected]

    @classmethod
    def _red_fraction(cls, image, circle):
        pixels = cls._disk_values(image, circle, outer=0.78).astype(np.float32)
        red = (pixels[:, 2] > pixels[:, 1] + 28) & (pixels[:, 2] > pixels[:, 0] + 28)
        return float(np.mean(red))

    @classmethod
    def _is_empty_source(cls, base_lab, current_lab, circle):
        background = np.median(cls._disk_values(base_lab, circle, inner=1.18, outer=1.52), axis=0)
        center = np.median(cls._disk_values(current_lab, circle, outer=0.72), axis=0)
        return float(np.linalg.norm(center - background)) <= 32.0

    @classmethod
    def _signature(cls, image, circle):
        patch = cls._circle_patch(image, circle)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ink = np.maximum(0.0, cv2.GaussianBlur(gray, (0, 0), 2.0) - gray)
        polar = cv2.warpPolar(
            ink, (12, 72), (23.5, 23.5), 18, cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR,
        )
        polar -= float(np.mean(polar))
        return polar

    @classmethod
    def _appearance_similarity(cls, base_image, current_image, source, destination):
        signature = cls._signature(base_image, source)
        norm_base = float(np.linalg.norm(signature))
        if norm_base < 12:
            return -1.0
        transformed = np.fft.rfft(signature, axis=0)
        best = -1.0
        for dx in (-0.12, 0.0, 0.12):
            for dy in (-0.12, 0.0, 0.12):
                x, y, _ = destination
                # 同一枚棋子的字形尺度保持一致，避免霍夫检测在内外圆边之间切换。
                radius = source[2]
                candidate = cls._signature(current_image, (x + dx * radius, y + dy * radius, radius))
                norm_current = float(np.linalg.norm(candidate))
                if norm_current < 12:
                    continue
                correlation = np.fft.irfft(
                    np.conj(transformed) * np.fft.rfft(candidate, axis=0), n=72, axis=0,
                ).sum(axis=1) / (norm_base * norm_current)
                best = max(best, float(np.max(correlation)))
        return best

    def detect(self, base_image, current_image, board, grid_mapper, rotated=False, ignored_squares=()):
        if base_image is None or current_image is None or base_image.shape != current_image.shape:
            return None, {'reason': '基准帧与当前帧尺寸不一致'}
        report = self.prepare_base(base_image, board, grid_mapper, rotated, ignored_squares)
        if not report['ready']:
            return None, report
        current = cv2.warpPerspective(current_image, self._baseline['transform'], self.SIZE)
        try:
            current, alignment = self._align(current)
        except (cv2.error, ValueError) as error:
            return None, {'reason': '画面配准待确认，请移开遮挡并核对定位', 'detail': str(error)}
        baseline = self._baseline
        report.update(alignment)
        circles = self._assign_circles(self._find_circles(current), baseline['positions'], 0.52)
        base_smoothed = cv2.GaussianBlur(baseline['image'], (5, 5), 0.9).astype(np.float32)
        current_smoothed = cv2.GaussianBlur(current, (5, 5), 0.9).astype(np.float32)
        differences = np.mean(np.abs(current_smoothed - base_smoothed), axis=2)
        base_lab = cv2.cvtColor(baseline['image'], cv2.COLOR_BGR2LAB).astype(np.float32)
        current_lab = cv2.cvtColor(current, cv2.COLOR_BGR2LAB).astype(np.float32)
        changed = {}
        for square, position in baseline['positions'].items():
            circle = baseline['circles'].get(square) or circles.get(square) or (*position, 21.0)
            changed[square] = float(np.mean(self._disk_values(differences, circle, outer=0.95)))
        lost = [square for square in baseline['circles']
                if square not in ignored_squares and square not in circles and changed[square] >= 4.0]
        report['vacated_squares'] = lost
        if len(lost) != 1:
            report['reason'] = '尚未确认单个棋子离开原格' if not lost else '检测到多个棋子区域消失，请移开人手并一次走一子'
            if len(lost) > 1 and all(self._is_empty_source(base_lab, current_lab, baseline['circles'][square]) for square in lost):
                report.update(illegal_move='multiple', illegal_squares=lost,
                              reason='走子错误，一次移动了多枚棋子，请恢复上一回合棋局')
            return None, report
        source_square = lost[0]
        source_circle = baseline['circles'][source_square]
        if not self._is_empty_source(base_lab, current_lab, source_circle):
            report['reason'] = f'源格 {source_square} 尚有遮挡，等待棋盘背景显露'
            return None, report

        moving_piece = baseline['pieces'][source_square]
        source_red = self._red_fraction(baseline['image'], source_circle)
        candidates = []
        for target, destination in circles.items():
            if target == source_square or changed[target] < 4.0:
                continue
            anchor = baseline['positions'][target]
            if math.hypot(destination[0] - anchor[0], destination[1] - anchor[1]) > self.CELL * 0.48:
                continue
            target_red = self._red_fraction(current, destination)
            if source_red >= 0.02 and target_red < 0.005:
                continue
            similarity = self._appearance_similarity(baseline['image'], current, source_circle, destination)
            target_piece = baseline['pieces'].get(target)
            if target_piece and target_piece.isupper() == moving_piece.isupper():
                original = baseline['circles'].get(target, (*anchor, source_circle[2]))
                if similarity < 0.65 or self._appearance_similarity(baseline['image'], current, original, destination) >= 0.8:
                    continue
            # 空格落子或黑子变为红子时，由离位源格确定棋子身份。
            previous_circle = baseline['circles'].get(target, (*anchor, source_circle[2]))
            red_increase = target_red - self._red_fraction(baseline['image'], previous_circle)
            red_arrival = target_red >= 0.005 and (
                target not in baseline['pieces'] or
                (baseline['pieces'][target].islower() and red_increase >= 0.005)
            )
            if similarity < 0.50 and not red_arrival:
                continue

            explained = np.zeros(differences.shape, np.uint8)
            known_motion = [(*baseline['positions'][square], self.CELL * 0.35) for square in ignored_squares]
            for x, y, radius in (source_circle, destination, *known_motion):
                cv2.circle(explained, (round(x), round(y)), round(max(radius * 1.7, self.CELL * 0.55)), 255, -1)
            unexplained = (differences > 16) & (explained == 0)
            unexplained[:24] = False
            unexplained[-24:] = False
            unexplained[:, :24] = False
            unexplained[:, -24:] = False
            outside = int(np.count_nonzero(unexplained))
            inside = int(np.count_nonzero((differences > 16) & (explained != 0)))
            if outside > max(self.CELL ** 2 * 0.8, inside * 0.65):
                continue
            score = similarity * 60 + min(changed[source_square], 40) + min(changed[target], 40)
            candidates.append((score, target, similarity, outside))

        if not candidates:
            report['reason'] = f'已确认 {source_square} 离位，目标格、棋子外观或走法仍待确认'
            return None, report
        candidates.sort(reverse=True)
        score, target, similarity, outside = candidates[0]
        second = candidates[1][0] if len(candidates) > 1 else 0.0
        margin = score - second
        if len(candidates) > 1 and margin < max(4.0, score * 0.08):
            report.update(reason='目标格存在竞争候选，等待后续画面确认', margin=round(margin, 2))
            return None, report
        standard_move = physical_square(source_square, rotated) + physical_square(target, rotated)
        if not board.is_legal_move(standard_move):
            report.update(
                reason=f'走子错误 {source_square}{target}，请恢复上一回合棋局',
                illegal_move=source_square + target, illegal_squares=[source_square, target],
            )
            return None, report
        report.update(
            reason='源格清空、目标占位与合法走法一致',
            best_move=source_square + target, piece=moving_piece,
            best_score=round(score, 2), margin=round(margin, 2),
            appearance_similarity=round(similarity, 3),
            unexplained_pixels=outside,
        )
        return (source_square, target), report
