"""用可重复的棋盘图像验证走子识别，覆盖炮、相、遮挡和帧确认。"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from xiangqi_engine import XiangqiBoard, square_to_coord
from xiangqi_vision import MoveConfirmation, XiangqiMoveDetector, physical_square
from xiangqi_agent import XiangqiAgent


class TestGrid:
    @staticmethod
    def get_board_bounding_rect_px(width, height):
        return 40, 40, 500, 550

    @staticmethod
    def get_square_norm_pos(square):
        column, row = square_to_coord(square)
        return (70 + column * 50) / 600 * 1000, (520 - row * 50) / 600 * 1000


def draw_board(board, rotated=False):
    image = np.full((600, 600, 3), (120, 160, 150), np.uint8)
    for column in range(9):
        cv2.line(image, (70 + column * 50, 70), (70 + column * 50, 520), (45, 65, 55), 1)
    for row in range(10):
        cv2.line(image, (70, 520 - row * 50), (470, 520 - row * 50), (45, 65, 55), 1)
    for row, rank in enumerate(board.board):
        for column, piece in enumerate(rank):
            if piece == '.':
                continue
            square = physical_square(chr(97 + column) + str(row), rotated)
            physical_column, physical_row = square_to_coord(square)
            x, y = 70 + physical_column * 50, 520 - physical_row * 50
            cv2.circle(image, (x, y), 17, (165, 215, 230), -1, cv2.LINE_AA)
            cv2.circle(image, (x, y), 17, (40, 45, 50), 2, cv2.LINE_AA)
            color = (20, 25, 190) if piece.isupper() else (25, 25, 25)
            cv2.putText(image, piece.upper(), (x - 7, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return image


class MoveVisionTest(unittest.TestCase):
    def setUp(self):
        self.board = XiangqiBoard()
        self.base = draw_board(self.board)
        self.grid = TestGrid()
        self.detector = XiangqiMoveDetector()

    def moved_image(self, move, rotated=False):
        board = XiangqiBoard(self.board.get_fen())
        self.assertTrue(board.apply_move(move)[0])
        return draw_board(board, rotated)

    def detect(self, image, rotated=False):
        return self.detector.detect(self.base, image, self.board, self.grid, rotated)

    def test_cannon_and_elephant_have_distinct_vacated_sources(self):
        for move, piece in [('b2e2', 'C'), ('c0e2', 'B'), ('g0e2', 'B'), ('a3a4', 'P')]:
            with self.subTest(move=move):
                result, info = self.detect(self.moved_image(move))
                self.assertEqual(result, (move[:2], move[2:]), info)
                self.assertEqual(info['piece'], piece)

    def test_all_initial_legal_moves_are_tracked(self):
        for move in self.board.generate_legal_moves():
            with self.subTest(move=move):
                result, info = self.detect(self.moved_image(move))
                self.assertEqual(result, (move[:2], move[2:]), info)

    def test_opposite_color_at_destination_is_rejected(self):
        board = XiangqiBoard()
        board.apply_move('b2e2')
        board.board[2][4] = 'b'
        result, info = self.detect(draw_board(board))
        self.assertIsNone(result, info)

    def test_new_destination_is_confirmed_when_glyph_match_is_weak(self):
        with patch.object(self.detector, '_appearance_similarity', return_value=0.3):
            result, info = self.detect(self.moved_image('b2b6'))
        self.assertEqual(result, ('b2', 'b6'), info)
        self.assertEqual(info['piece'], 'C')

    def test_refinement_tolerates_piece_center_scatter(self):
        positions = self.detector._nominal_positions()
        squares = [chr(97 + column) + str(row)
                   for row, rank in enumerate(self.board.board)
                   for column, piece in enumerate(rank) if piece != '.']
        expected = np.float32([positions[square] for square in squares])
        adjusted = expected * 0.94 + [12.0, 9.0]
        measured = adjusted + np.random.default_rng(15).uniform(-13.0, 13.0, expected.shape)
        observed = {square: (*point, 21.0) for square, point in zip(squares, measured)}
        refined = self.detector._refine_positions(positions, observed)
        refined_points = np.float32([refined[square] for square in squares])
        self.assertLess(float(np.mean(np.linalg.norm(refined_points - adjusted, axis=1))), 8.0)

    def test_capture_is_confirmed_when_glyph_match_is_weak(self):
        self.board = XiangqiBoard('3k5/9/1p7/9/9/1P7/9/1C7/9/4K4 w - - 0 1')
        self.base = draw_board(self.board)
        with patch.object(self.detector, '_appearance_similarity', return_value=0.3):
            result, info = self.detect(self.moved_image('b2b7'))
        self.assertEqual(result, ('b2', 'b7'), info)
        self.assertEqual(info['piece'], 'C')

    def test_baseline_counts_drop_after_red_and_black_capture(self):
        for move in ('b2b4', 'c6c5', 'b4e4', 'g9e7', 'e4e7', 'b9c7', 'a3a4', 'b7e7'):
            ok, captured = self.board.apply_move(move)
            self.assertTrue(ok, move)
            if captured:
                expected = 31 if captured == 'b' else 30
                report = self.detector.prepare_base(draw_board(self.board), self.board, self.grid)
                self.assertEqual(report['expected_pieces'], expected)
                self.assertEqual(report['visible_pieces'], expected, report)
                self.assertTrue(report['ready'], report)

    def test_position_recovery_finds_capture_without_reference(self):
        for move in ('b2b4', 'c6c5', 'b4e4', 'g9e7'):
            self.assertTrue(self.board.apply_move(move)[0])
        after = XiangqiBoard(self.board.get_fen())
        self.assertTrue(after.apply_move('e4e7')[0])
        for rotated in (False, True):
            with self.subTest(rotated=rotated):
                move, info = self.detector.detect_from_position(
                    draw_board(after, rotated), self.board, self.grid, rotated)
                expected = ('e5', 'e2') if rotated else ('e4', 'e7')
                self.assertEqual(move, expected, info)
                self.assertEqual(info['captured'], 'b')
                self.assertEqual(info['piece_count_after_move'], 31)

    def test_position_recovery_requires_destination(self):
        current = XiangqiBoard(self.board.get_fen())
        current.board[2][1] = '.'
        move, info = self.detector.detect_from_position(draw_board(current), self.board, self.grid)
        self.assertIsNone(move, info)

    def test_position_recovery_waits_for_hand_to_leave(self):
        current = self.moved_image('b2e2')
        cv2.rectangle(current, (99, 399), (141, 441), (100, 150, 215), -1)
        move, info = self.detector.detect_from_position(current, self.board, self.grid)
        self.assertIsNone(move, info)

    def test_black_capture_baseline_recovers_red_circle_miss(self):
        for move in ('b2b4', 'c6c5', 'b4e4', 'g9e7', 'e4e7', 'b9c7', 'a3a4', 'b7e7'):
            self.assertTrue(self.board.apply_move(move)[0])
        find = self.detector._find_circles
        x, y = self.detector._nominal_positions()['c0']

        def miss_one_red(image):
            return [circle for circle in find(image)
                    if np.hypot(circle[0] - x, circle[1] - y) > self.detector.CELL * 0.5]

        with patch.object(self.detector, '_find_circles', side_effect=miss_one_red):
            report = self.detector.prepare_base(draw_board(self.board), self.board, self.grid)
        self.assertTrue(report['ready'], report)
        self.assertEqual(report['expected_pieces'], 30)
        self.assertIn('c0', report['recovered_red_squares'])

    def test_normalized_geometry_handles_frame_sizes(self):
        original = self.base
        moved = self.moved_image('b2e2')
        for size in [(300, 300), (900, 500), (1280, 720)]:
            with self.subTest(size=size):
                self.base = cv2.resize(original, size, interpolation=cv2.INTER_AREA)
                current = cv2.resize(moved, size, interpolation=cv2.INTER_AREA)
                result, info = self.detect(current)
                self.assertEqual(result, ('b2', 'e2'), info)

    def test_stationary_elephant_with_lighting_change_is_not_source(self):
        current = self.moved_image('b2e2')
        patch = current[493:548, 143:198].astype(np.int16)
        current[493:548, 143:198] = np.clip(patch + 16, 0, 255).astype(np.uint8)
        result, info = self.detect(current)
        self.assertEqual(result, ('b2', 'e2'), info)

    def test_cannon_move_with_glare_on_stationary_elephant_keeps_correct_source(self):
        current = self.moved_image('b2e2')
        for image in [self.base, current]:
            face = np.all(image == np.array([165, 215, 230], np.uint8), axis=2)
            image[face] = [130, 170, 160]
        yy, xx = np.ogrid[:600, :600]
        glare = (xx - 170) ** 2 + (yy - 520) ** 2 <= 21 ** 2
        current[glare] = np.clip(current[glare].astype(np.int16) + 90, 0, 255)
        # 相留在 c0，旧差分算法在此画面上错误选择 c0e2。
        result, info = self.detect(current)
        self.assertEqual(result, ('b2', 'e2'), info)
        self.assertEqual(info['piece'], 'C')

    def test_shadow_without_piece_movement_is_rejected(self):
        current = self.base.copy()
        current[490:548, 145:198] = np.clip(current[490:548, 145:198].astype(np.int16) + 25, 0, 255)
        result, info = self.detect(current)
        self.assertIsNone(result, info)

    def test_unchanged_board_and_global_light_changes_are_rejected(self):
        for delta in [0, 30, -25]:
            with self.subTest(delta=delta):
                current = np.clip(self.base.astype(np.int16) + delta, 0, 255).astype(np.uint8)
                self.assertIsNone(self.detect(current)[0])

    def test_small_camera_shift_is_aligned(self):
        transform = np.float32([[1, 0, 2], [0, 1, -1]])
        unchanged = cv2.warpAffine(self.base, transform, (600, 600), borderMode=cv2.BORDER_REFLECT_101)
        self.assertIsNone(self.detect(unchanged)[0])
        moved = cv2.warpAffine(self.moved_image('b2e2'), transform, (600, 600), borderMode=cv2.BORDER_REFLECT_101)
        result, info = self.detect(moved)
        self.assertEqual(result, ('b2', 'e2'), info)

    def test_large_camera_shift_is_rejected(self):
        transform = np.float32([[1, 0, 25], [0, 1, 0]])
        moved = cv2.warpAffine(self.moved_image('b2e2'), transform, (600, 600), borderMode=cv2.BORDER_REFLECT_101)
        self.assertIsNone(self.detect(moved)[0])

    def test_two_piece_movements_are_rejected(self):
        board = XiangqiBoard()
        board.board[2][4] = board.board[2][1]
        board.board[2][1] = '.'
        board.board[2][0] = board.board[0][2]
        board.board[0][2] = '.'
        result, info = self.detect(draw_board(board))
        self.assertIsNone(result, info)

    def test_pickup_without_placement_is_rejected(self):
        board = XiangqiBoard()
        board.board[2][1] = '.'
        result, info = self.detect(draw_board(board))
        self.assertIsNone(result, info)

    def test_illegal_cannon_diagonal_is_rejected(self):
        board = XiangqiBoard()
        board.board[4][2] = board.board[2][1]
        board.board[2][1] = '.'
        move, info = self.detect(draw_board(board))
        self.assertIsNone(move)
        self.assertEqual(info['illegal_move'], 'b2c4')

    def test_moving_black_piece_on_red_turn_is_reported(self):
        current = XiangqiBoard(self.board.get_fen())
        current.board[5][0] = current.board[6][0]
        current.board[6][0] = '.'
        move, info = self.detect(draw_board(current))
        self.assertIsNone(move)
        self.assertEqual(info['illegal_move'], 'a6a5')

    def test_position_recovery_reports_illegal_move(self):
        current = XiangqiBoard(self.board.get_fen())
        current.board[4][2] = current.board[2][1]
        current.board[2][1] = '.'
        move, info = self.detector.detect_from_position(draw_board(current), self.board, self.grid)
        self.assertIsNone(move)
        self.assertEqual(info['illegal_move'], 'b2c4')

    def test_undo_requires_captured_piece_to_be_returned(self):
        current = XiangqiBoard(self.board.get_fen())
        current.board[0][2] = '.'
        restored, info = self.detector.matches_snapshot(
            self.base, draw_board(current), self.board, self.grid, affected_squares=('b2','c0'))
        self.assertFalse(restored, info)
        restored, info = self.detector.matches_snapshot(
            self.base, self.base.copy(), self.board, self.grid, affected_squares=('b2','c0'))
        self.assertTrue(restored, info)

    def test_capture_keeps_destination_occupied(self):
        self.board = XiangqiBoard('3k5/9/1p7/9/9/1P7/9/1C7/9/4K4 w - - 0 1')
        self.base = draw_board(self.board)
        result, info = self.detect(self.moved_image('b2b7'))
        self.assertEqual(result, ('b2', 'b7'), info)

    def test_rotated_board_returns_physical_squares(self):
        self.base = draw_board(self.board, rotated=True)
        result, info = self.detect(self.moved_image('b2e2', rotated=True), rotated=True)
        self.assertEqual(result, ('h7', 'e7'), info)
        self.assertEqual(info['piece'], 'C')

    def test_partial_hand_occlusion_is_rejected(self):
        current = self.moved_image('b2e2')
        cv2.rectangle(current, (135, 360), (235, 460), (100, 150, 215), -1)
        result, info = self.detect(current)
        self.assertIsNone(result, info)

    def test_baseline_with_large_occlusion_is_rejected(self):
        cv2.rectangle(self.base, (40, 40), (495, 225), (215, 215, 215), -1)
        report = self.detector.prepare_base(self.base, self.board, self.grid)
        self.assertFalse(report['ready'])

    def test_missing_red_cannon_prevents_incomplete_baseline(self):
        cv2.rectangle(self.base, (99, 399), (141, 441), (215, 215, 215), -1)
        report = self.detector.prepare_base(self.base, self.board, self.grid)
        self.assertFalse(report['ready'])
        self.assertIn('b2', report['missing_red_squares'])

    def test_rotated_glyph_is_matched(self):
        current = self.moved_image('b2e2')
        patch = current[396:444, 246:294].copy()
        transform = cv2.getRotationMatrix2D((23.5, 23.5), 70, 1.0)
        current[396:444, 246:294] = cv2.warpAffine(patch, transform, (48, 48), borderMode=cv2.BORDER_REFLECT_101)
        result, info = self.detect(current)
        self.assertEqual(result, ('b2', 'e2'), info)


class MoveConfirmationTest(unittest.TestCase):
    def test_repeated_frame_does_not_confirm(self):
        confirmation = MoveConfirmation()
        for now in [0.0, 0.2, 0.5, 0.8]:
            self.assertFalse(confirmation.observe(('b2', 'e2'), frame_key=1, now=now))
        self.assertEqual(confirmation.count, 1)

    def test_three_distinct_stable_frames_confirm(self):
        confirmation = MoveConfirmation()
        self.assertFalse(confirmation.observe(('b2', 'e2'), 1, now=0.0))
        self.assertFalse(confirmation.observe(('b2', 'e2'), 2, now=0.2))
        self.assertTrue(confirmation.observe(('b2', 'e2'), 3, now=0.4))

    def test_changed_candidate_or_occlusion_resets_confirmation(self):
        confirmation = MoveConfirmation()
        confirmation.observe(('b2', 'e2'), 1, now=0.0)
        confirmation.observe(('b2', 'e2'), 2, now=0.2)
        self.assertFalse(confirmation.observe(('c0', 'e2'), 3, now=0.4))
        self.assertEqual(confirmation.count, 1)
        confirmation.observe(None, 4, now=0.6)
        self.assertEqual(confirmation.count, 0)


class AgentMoveIntegrationTest(unittest.TestCase):
    def _run_illegal_turn(self, restore=False, retry_legal=False):
        board = XiangqiBoard()
        base = draw_board(board)
        wrong = XiangqiBoard(board.get_fen())
        wrong.board[4][2] = wrong.board[2][1]
        wrong.board[2][1] = '.'
        illegal_image = draw_board(wrong)
        corrected = XiangqiBoard(board.get_fen())
        self.assertTrue(corrected.apply_move('b2e2')[0])
        corrected_image = draw_board(corrected)
        clock = SimpleNamespace(now=0.0, calls=0, frame=0)
        seen_states = []

        def provider():
            clock.frame += 1
            seen_states.append(agent.current_state)
            image = illegal_image
            if agent.current_state == 'WAIT_BASE':
                image = base
            elif agent.current_state == 'WAIT_UNDO':
                if restore:
                    image = base
                elif retry_legal:
                    image = corrected_image
            return image.copy(), {'fresh': True, 'frame_id': clock.frame}

        driver = MagicMock()
        engine = MagicMock()
        agent = XiangqiAgent(driver, MagicMock(), TestGrid(), frame_provider=provider, pikafish=engine)
        agent.is_running = True
        agent.current_state = 'WAIT_BASE'
        agent._baseline_home_ready = True

        def sleep(seconds, step=0.05):
            clock.now += seconds
            clock.calls += 1
            if clock.calls > 24 or agent.status_text == '已恢复上一回合棋局，请重新走棋':
                agent.is_running = False
            return agent.is_running

        with patch.object(agent, '_sleep_cancelable', side_effect=sleep), \
                patch('xiangqi_agent.time.time', side_effect=lambda: clock.now), \
                patch('xiangqi_agent.time.monotonic', side_effect=lambda: clock.now):
            agent._battle_loop(agent.game_id)
        self.assertIn('WAIT_UNDO', seen_states)
        self.assertEqual(agent.board.get_fen(), board.get_fen())
        self.assertEqual(agent._turn_snapshot['fen'], board.get_fen())
        self.assertTrue(np.array_equal(agent._turn_snapshot['frame'], base))
        self.assertEqual(agent.move_history, [])
        engine.get_best_move.assert_not_called()
        driver.send_angles.assert_not_called()
        driver.send_coords.assert_not_called()
        return agent

    def test_illegal_move_waits_for_previous_turn_snapshot(self):
        agent = self._run_illegal_turn()
        self.assertEqual(agent.current_state, 'WAIT_UNDO')
        self.assertIn('走子错误', agent.status_text)

    def test_restored_snapshot_resumes_red_turn(self):
        agent = self._run_illegal_turn(restore=True)
        self.assertEqual(agent.current_state, 'MONITOR')
        self.assertIsNone(agent._illegal_move)

    def test_changing_to_legal_move_still_requires_restore_first(self):
        agent = self._run_illegal_turn(retry_legal=True)
        self.assertEqual(agent.current_state, 'WAIT_UNDO')

    def _run_capture_during_baseline_sampling(self, use_reference=True):
        board = XiangqiBoard()
        for move in ('b2b4', 'c6c5', 'b4e4'):
            self.assertTrue(board.apply_move(move)[0])
        reference = draw_board(board)
        after_capture = XiangqiBoard(board.get_fen())
        for move in ('g9e7', 'e4e7'):
            self.assertTrue(after_capture.apply_move(move)[0])
        current = draw_board(after_capture)
        clock = SimpleNamespace(now=0.0, calls=0, frame=0)

        def provider():
            clock.frame += 1
            return current.copy(), {'fresh': True, 'frame_id': clock.frame}

        driver = MagicMock()
        engine = MagicMock()
        agent = XiangqiAgent(driver, MagicMock(), TestGrid(), frame_provider=provider, pikafish=engine)
        agent.board = board
        agent.base_frame = reference if use_reference else None
        agent.is_running = True
        agent.last_black_move = 'g9e7'
        agent._last_black_std_move = 'g9e7'
        with patch.object(agent, '_physical_move_piece'):
            agent._execute_arm_move_phases(agent.game_id, agent.board_version)
        agent.current_state = 'WAIT_BASE'
        agent._baseline_home_ready = True

        def sleep(seconds, step=0.05):
            clock.now += seconds
            clock.calls += 1
            if clock.calls > 12:
                agent.is_running = False
            return agent.is_running

        def think(fen, movetime_ms):
            agent.is_running = False
            return 'a6a5', None

        engine.get_best_move.side_effect = think
        with patch.object(agent, '_sleep_cancelable', side_effect=sleep), \
                patch.object(agent.move_detector, '_appearance_similarity', return_value=0.3), \
                patch('xiangqi_agent.time.time', side_effect=lambda: clock.now), \
                patch('xiangqi_agent.time.monotonic', side_effect=lambda: clock.now):
            agent._battle_loop(agent.game_id)

        self.assertEqual(engine.get_best_move.call_count, 1)
        self.assertEqual(agent.last_red_move, 'e4e7')
        self.assertEqual(agent.move_history[-1]['captured'], 'b')
        self.assertEqual(agent.board.get_fen(), after_capture.get_fen())
        self.assertEqual(agent.last_detection_info['confirmation_frames'], 3)
        report = agent.move_detector.prepare_base(current, agent.board, TestGrid())
        self.assertTrue(report['ready'], report)
        self.assertEqual(report['expected_pieces'], 31)
        driver.send_angles.assert_not_called()
        driver.send_coords.assert_not_called()

    def test_capture_during_baseline_sampling_updates_board_and_resumes(self):
        self._run_capture_during_baseline_sampling()

    def test_capture_without_reference_resumes_after_frame_confirmation(self):
        self._run_capture_during_baseline_sampling(use_reference=False)

    def test_black_capture_discards_to_p5_before_committing_board(self):
        board = XiangqiBoard()
        for move in ('b2b4', 'c6c5', 'b4e4', 'g9e7', 'e4e7', 'b9c7', 'a3a4'):
            self.assertTrue(board.apply_move(move)[0])
        agent = XiangqiAgent(MagicMock(), MagicMock(), TestGrid(), pikafish=MagicMock())
        agent.board = board
        agent.is_running = True
        agent.last_black_move = 'b7e7'
        agent._last_black_std_move = 'b7e7'
        with patch.object(agent, '_physical_pick_and_place_square') as discard, \
                patch.object(agent, '_physical_move_piece') as move, \
                patch.object(agent, '_sleep_cancelable', return_value=True):
            agent._execute_arm_move_phases(agent.game_id, agent.board_version)
        discard.assert_called_once_with('e7', dest_station='P5', cancel_token=None)
        move.assert_called_once_with('b7', 'e7', cancel_token=None)
        self.assertEqual(agent.move_history[-1]['captured'], 'C')
        self.assertEqual(agent._turn_snapshot['fen'], agent.board.get_fen())
        report = agent.move_detector.prepare_base(draw_board(agent.board), agent.board, TestGrid())
        self.assertTrue(report['ready'], report)
        self.assertEqual(report['expected_pieces'], 30)

    def run_red_move(self, frame_ids):
        driver = MagicMock()
        engine = MagicMock()
        board = XiangqiBoard()
        base = draw_board(board)
        self.assertTrue(board.apply_move('b2e2')[0])
        current = draw_board(board)
        self.assertLess(float(np.mean(cv2.absdiff(base, current))), 4.5)
        clock = SimpleNamespace(now=0.0, calls=0)
        sequence = iter(frame_ids)
        last_id = frame_ids[-1]

        def provider():
            return current.copy(), {'fresh': True, 'frame_id': next(sequence, last_id)}

        agent = XiangqiAgent(driver, MagicMock(), TestGrid(), frame_provider=provider, pikafish=engine)
        agent.base_frame = base
        agent.prev_frame = current.copy()
        agent.is_running = True
        agent.current_state = 'MONITOR'
        agent.stable_duration = 1.2
        agent.last_check_time = 0.0

        def sleep(seconds, step=0.05):
            clock.now += seconds
            clock.calls += 1
            if clock.calls > 12:
                agent.is_running = False
            return agent.is_running

        def think(fen, movetime_ms):
            agent.is_running = False
            return 'a6a5', None

        engine.get_best_move.side_effect = think
        with patch.object(agent, '_sleep_cancelable', side_effect=sleep), \
                patch('xiangqi_agent.time.time', side_effect=lambda: clock.now), \
                patch('xiangqi_agent.time.monotonic', side_effect=lambda: clock.now):
            agent._battle_loop(agent.game_id)
        driver.send_angles.assert_not_called()
        driver.send_coords.assert_not_called()
        return agent, engine

    def test_normal_loop_commits_cannon_after_distinct_frames(self):
        agent, engine = self.run_red_move([0, 1, 1, 2, 3])
        self.assertEqual(engine.get_best_move.call_count, 1)
        self.assertEqual(agent.last_red_move, 'b2e2')
        self.assertEqual(agent.board.get_piece('c0'), 'B')
        self.assertEqual(agent.board.get_piece('e2'), 'C')
        self.assertEqual(agent.move_history[-1]['piece_name'], '红炮')
        self.assertEqual(agent.last_detection_info['confirmation_frames'], 3)

    def test_repeated_camera_frame_does_not_commit_board(self):
        agent, engine = self.run_red_move([7, 7, 7, 7])
        engine.get_best_move.assert_not_called()
        self.assertIsNone(agent.last_red_move)
        self.assertEqual(agent.board.get_piece('b2'), 'C')
        self.assertEqual(agent.board.get_piece('c0'), 'B')

    def test_occluded_baseline_retries_frames_without_repeating_home(self):
        driver = MagicMock()
        image = draw_board(XiangqiBoard())
        cv2.rectangle(image, (40, 40), (495, 225), (215, 215, 215), -1)
        agent = XiangqiAgent(
            driver, MagicMock(), TestGrid(),
            frame_provider=lambda: (image, {'fresh': True, 'frame_id': 1}), pikafish=MagicMock(),
        )
        agent.is_running = True
        agent.current_state = 'WAIT_BASE'
        calls = []

        def sleep(seconds, step=0.05):
            calls.append(seconds)
            if len(calls) >= 4:
                agent.is_running = False
            return agent.is_running

        with patch.object(agent, '_sleep_cancelable', side_effect=sleep):
            agent._battle_loop(agent.game_id)
        self.assertEqual(driver.send_angles.call_count, 1)
        self.assertIsNone(agent.base_frame)
        self.assertFalse(agent.last_detection_info['ready'])


if __name__ == '__main__':
    unittest.main()
