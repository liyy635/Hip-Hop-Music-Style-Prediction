import argparse
import os
import sys
from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
	sys.path.insert(0, SCRIPT_DIR)

from Train import AudioEncoder, DEFAULT_MUSIC_ROOT, _get_checkpoint_path

TOOL_DIR = os.path.join(BASE_DIR, "Tool")
if TOOL_DIR not in sys.path:
	sys.path.insert(0, TOOL_DIR)


AUDIO_EXTENSIONS = {
	".mp3",
	".wav",
	".flac",
	".m4a",
	".aac",
	".ogg",
	".wma",
	".mp4",
	".mkv",
}


def load_checkpoint_2d(device: torch.device, data_root: str) -> Dict:
	ckpt_path = _get_checkpoint_path(data_root)
	if not os.path.isfile(ckpt_path):
		raise FileNotFoundError(f"[2D-Test] 找不到 checkpoint 文件: {ckpt_path}\n请先运行 Train.py 进行训练。")
	return torch.load(ckpt_path, map_location=device)


def _preprocess_single_audio_2d(
	audio_path: str,
	sr: int,
	segment_duration: float,
	n_mels: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	if not os.path.isfile(audio_path):
		raise FileNotFoundError(f"音频文件不存在: {audio_path}")

	y, _ = librosa.load(audio_path, sr=sr, mono=True)
	target_len = int(segment_duration * sr)
	if len(y) >= target_len:
		start = max(0, (len(y) - target_len) // 2)
		y = y[start : start + target_len]
	else:
		y = np.pad(y, (0, target_len - len(y)), mode="constant")

	max_val = np.max(np.abs(y)) if len(y) > 0 else 0.0
	if max_val > 0:
		y = y / max_val

	harm, perc = librosa.effects.hpss(y)

	def _to_logmel(x: np.ndarray) -> torch.Tensor:
		S = librosa.feature.melspectrogram(
			y=x,
			sr=sr,
			n_mels=n_mels,
			fmin=20.0,
			fmax=sr / 2.0,
			n_fft=1024,
			hop_length=512,
			power=2.0,
		)
		S_db = librosa.power_to_db(S, ref=np.max)
		mu = float(np.mean(S_db))
		sigma = float(np.std(S_db)) + 1e-6
		S_norm = (S_db - mu) / sigma
		return torch.from_numpy(S_norm).unsqueeze(0).unsqueeze(0).float()

	return _to_logmel(perc), _to_logmel(harm), _to_logmel(y)


def _infer_probs_for_audio(
	audio_path: str,
	state: Dict,
	drum_encoder: AudioEncoder,
	timbre_encoder: AudioEncoder,
	jazz_encoder: AudioEncoder | None,
	device: torch.device,
	repeats: int = 10,
) -> np.ndarray:
	"""重复推理多次，平均每个风格的概率。"""

	class_names: List[str] = list(state["class_names"])
	sr: int = int(state["sr"])
	n_mels: int = int(state["n_mels"])
	segment_duration: float = float(state["segment_duration"])
	alpha_drum_global: float = float(state.get("alpha_drum", 0.5))
	alpha_drum_global = float(max(0.0, min(1.0, alpha_drum_global)))

	prototypes_drum: torch.Tensor = state["prototypes_drum"].to(device)
	prototypes_timbre: torch.Tensor = state["prototypes_timbre"].to(device)
	prototype_jazz_1d = state.get("prototype_jazz_1d", None)
	jazz_style_idx = int(state.get("jazz_style_idx", -1))
	if prototype_jazz_1d is not None:
		prototype_jazz_1d = prototype_jazz_1d.to(device)

	alpha_drum_per_class: torch.Tensor | None = None
	if "alpha_drum_per_class" in state:
		alpha_drum_per_class = state["alpha_drum_per_class"].to(device).view(-1)
		if alpha_drum_per_class.shape[0] != len(class_names):
			alpha_drum_per_class = None

	score_runs: List[np.ndarray] = []
	for _ in range(repeats):
		feat_perc, feat_harm, feat_full = _preprocess_single_audio_2d(
			audio_path=audio_path,
			sr=sr,
			segment_duration=segment_duration,
			n_mels=n_mels,
		)
		feat_perc = feat_perc.to(device)
		feat_harm = feat_harm.to(device)
		feat_full = feat_full.to(device)

		with torch.no_grad():
			emb_drum = F.normalize(drum_encoder(feat_perc), dim=1)
			emb_timbre = F.normalize(timbre_encoder(feat_harm), dim=1)
			sims_drum = torch.matmul(emb_drum, prototypes_drum.T).squeeze(0)
			sims_timbre = torch.matmul(emb_timbre, prototypes_timbre.T).squeeze(0)

			if alpha_drum_per_class is None:
				alpha_eff = alpha_drum_global
				alpha_timbre_eff = 1.0 - alpha_eff
				score = alpha_eff * sims_drum + alpha_timbre_eff * sims_timbre
			else:
				alpha_vec = torch.clamp(alpha_drum_per_class, 0.0, 1.0)
				score = alpha_vec * sims_drum + (1.0 - alpha_vec) * sims_timbre

			if (
				jazz_encoder is not None
				and prototype_jazz_1d is not None
				and 0 <= jazz_style_idx < len(class_names)
			):
				emb_jazz = F.normalize(jazz_encoder(feat_full), dim=1)
				sim_jazz = torch.matmul(emb_jazz, prototype_jazz_1d.view(-1, 1)).squeeze()
				score[jazz_style_idx] = sim_jazz

		probs = torch.softmax(score, dim=0).cpu().numpy().astype(float)
		score_runs.append(probs)

	probs_runs = np.stack(score_runs, axis=0)
	if probs_runs.shape[0] <= 2:
		avg_probs = probs_runs.mean(axis=0)
	else:
		avg_probs = np.sort(probs_runs, axis=0)[1:-1].mean(axis=0)

	sum_probs = float(avg_probs.sum())
	if sum_probs > 0:
		avg_probs = avg_probs / sum_probs
	return avg_probs


def _split_excluded_styles(raw: str) -> List[str]:
	if not raw:
		return []
	return [x.strip() for x in raw.split(",") if x.strip()]


def _is_audio_file(file_name: str) -> bool:
	return os.path.splitext(file_name)[1].lower() in AUDIO_EXTENSIONS


def _ask_text(prompt: str) -> str:
	print(prompt, flush=True)
	return input("> ").strip()


def _ask_int(prompt: str, default: int) -> int:
	print(prompt, flush=True)
	val = input("> ").strip()
	if not val:
		return default
	try:
		return int(val)
	except ValueError:
		print("输入无效, 使用默认值")
		return default


def _ask_float(prompt: str, default: float) -> float:
	print(prompt, flush=True)
	val = input("> ").strip()
	if not val:
		return default
	try:
		return float(val)
	except ValueError:
		print("输入无效, 使用默认值")
		return default


def _format_confidence(avg_probs: np.ndarray) -> Tuple[float, float, float, str]:
	if avg_probs.size == 0:
		return 0.0, 0.0, 0.0, "low"
	top2 = np.sort(avg_probs)[::-1][:2]
	top1_prob = float(top2[0])
	top2_prob = float(top2[1]) if top2.size > 1 else 0.0
	prob_margin = top1_prob - top2_prob
	confidence = 0.5 * top1_prob + 0.5 * max(0.0, prob_margin)
	if confidence >= 0.75:
		level = "high"
	elif confidence >= 0.55:
		level = "medium"
	else:
		level = "low"
	return top1_prob, top2_prob, confidence, level


def predict_directory(
	target_dir: str,
	data_root: str = DEFAULT_MUSIC_ROOT,
	output_txt: str | None = None,
	top_k: int = 1,
	repeats: int = 10,
	excluded_styles: List[str] | None = None,
) -> str:
	if not os.path.isdir(target_dir):
		raise FileNotFoundError(f"要预测的文件夹不存在: {target_dir}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	state = load_checkpoint_2d(device, data_root)
	class_names: List[str] = list(state["class_names"])
	name_to_idx = {name: idx for idx, name in enumerate(class_names)}

	excluded_styles = excluded_styles or []
	excluded_set = {x.strip() for x in excluded_styles if x.strip()}
	unknown_styles = sorted([name for name in excluded_set if name not in name_to_idx])
	if unknown_styles:
		raise ValueError(f"以下风格不在 checkpoint 类别中: {unknown_styles}")
	excluded_indices = {name_to_idx[name] for name in excluded_set}
	active_indices = [i for i in range(len(class_names)) if i not in excluded_indices]
	if not active_indices:
		raise ValueError("所有风格都被排除了, 无法进行预测。")
	if top_k < 1 or top_k > len(active_indices):
		raise ValueError(f"top_k 必须在 [1, {len(active_indices)}] 之间")

	drum_encoder = AudioEncoder(n_mels=int(state["n_mels"]), embed_dim=int(state["embed_dim"])).to(device)
	timbre_encoder = AudioEncoder(n_mels=int(state["n_mels"]), embed_dim=int(state["embed_dim"])).to(device)
	drum_encoder.load_state_dict(state["drum_encoder_state_dict"])
	timbre_encoder.load_state_dict(state["timbre_encoder_state_dict"])
	drum_encoder.eval()
	timbre_encoder.eval()
	jazz_encoder = None
	if (
		"jazz_encoder_state_dict" in state
		and state.get("prototype_jazz_1d", None) is not None
		and 0 <= int(state.get("jazz_style_idx", -1)) < len(class_names)
	):
		jazz_encoder = AudioEncoder(n_mels=int(state["n_mels"]), embed_dim=int(state["embed_dim"])).to(device)
		jazz_encoder.load_state_dict(state["jazz_encoder_state_dict"])
		jazz_encoder.eval()

	if output_txt:
		output_path = output_txt
	else:
		folder_name = os.path.basename(os.path.normpath(target_dir)) or "result"
		output_path = os.path.join(target_dir, f"{folder_name}_predict_result.txt")

	folder_name = os.path.basename(os.path.normpath(target_dir)) or "result"
	default_file_name = f"{folder_name}_predict_result.txt"
	if os.path.isdir(output_path):
		output_path = os.path.join(output_path, default_file_name)

	output_parent = os.path.dirname(output_path)
	if output_parent:
		os.makedirs(output_parent, exist_ok=True)

	audio_files = [
		os.path.join(target_dir, file_name)
		for file_name in sorted(os.listdir(target_dir))
		if os.path.isfile(os.path.join(target_dir, file_name)) and _is_audio_file(file_name)
	]
	if not audio_files:
		raise FileNotFoundError("指定文件夹中没有找到任何支持格式的音频文件。")

	lines: List[str] = []
	lines.append("==== 2D 文件夹预测结果 ====")
	lines.append(f"时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	lines.append(f"预测文件夹: {target_dir}")
	lines.append(f"数据根目录: {data_root}")
	lines.append(f"Top-K: {top_k}")
	lines.append(f"每首歌重复次数: {repeats}")
	lines.append(f"排除风格: {sorted(excluded_set) if excluded_set else '无'}")
	lines.append(f"音频文件数: {len(audio_files)}")
	lines.append("")

	print("\n[2D-Test] ==== 文件夹批量预测 ====")
	print(f"[2D-Test] 预测文件夹: {target_dir}")
	print(f"[2D-Test] 输出文件: {output_path}")
	print(f"[2D-Test] 音频文件数: {len(audio_files)}")

	for index, audio_path in enumerate(audio_files, start=1):
		file_name = os.path.basename(audio_path)
		try:
			avg_probs = _infer_probs_for_audio(
				audio_path=audio_path,
				state=state,
				drum_encoder=drum_encoder,
				timbre_encoder=timbre_encoder,
				jazz_encoder=jazz_encoder,
				device=device,
				repeats=repeats,
			)
			pred_score = avg_probs.copy()
			if excluded_indices:
				pred_score[list(excluded_indices)] = -np.inf
			pred_indices = [int(x) for x in np.argsort(-pred_score)[:top_k]]
			pred_names = [class_names[i] for i in pred_indices]
			top1_prob, top2_prob, confidence, level = _format_confidence(avg_probs)
			line = "\t".join([
				file_name,
				f"Top1: {pred_names[0]}",
				f"Top{top_k}: {', '.join(pred_names)}",
				f"conf={confidence:.4f}",
				f"level={level}",
				f"top1_prob={top1_prob:.4f}",
				f"top2_prob={top2_prob:.4f}",
			])
			print(f"[2D-Test] [{index}/{len(audio_files)}] {line}")
			lines.append(line)
		except Exception as e:
			line = f"{file_name}\tERROR: {e}"
			print(f"[2D-Test] [{index}/{len(audio_files)}] {line}")
			lines.append(line)

	try:
		with open(output_path, "w", encoding="utf-8") as f:
			f.write("\n".join(lines) + "\n")
	except PermissionError as e:
		raise PermissionError(
			f"无法写入输出文件: {output_path}. 如果你输入的是目录路径, 请改为具体的 .txt 文件路径, "
			f"或直接回车使用默认输出文件。原始错误: {e}"
		) from e

	print(f"[2D-Test] 已写入结果文件: {output_path}")
	return output_path


def run_test(
	data_root: str,
	top_k: int = 1,
	rounds: int = 1,
	repeats_per_sample: int = 10,
	excluded_styles: List[str] | None = None,
) -> None:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	state = load_checkpoint_2d(device, data_root)
	class_names: List[str] = list(state["class_names"])
	num_classes = len(class_names)

	excluded_styles = excluded_styles or []
	excluded_set = {x.strip() for x in excluded_styles if x.strip()}
	name_to_idx = {name: idx for idx, name in enumerate(class_names)}
	unknown_styles = sorted([name for name in excluded_set if name not in name_to_idx])
	if unknown_styles:
		raise ValueError(f"以下风格不在 checkpoint 类别中: {unknown_styles}")

	excluded_indices = {name_to_idx[name] for name in excluded_set}
	active_indices = [i for i in range(num_classes) if i not in excluded_indices]
	active_set = set(active_indices)
	if not active_indices:
		raise ValueError("所有风格都被排除了, 无法进行测试。")

	test_samples = state.get("test_samples", None)
	if not test_samples:
		raise ValueError("[2D-Test] checkpoint 中没有保存 test_samples, 请使用新版 Train.py 重新训练以生成训练/测试划分。")

	if top_k < 1 or top_k > len(active_indices):
		raise ValueError(f"top_k 必须在 [1, {len(active_indices)}] 之间 (当前参与预测的风格数)")

	num_test_samples = sum(1 for _, label_idx in test_samples if int(label_idx) in active_set)
	if num_test_samples == 0:
		raise ValueError("测试集中没有可参与测试的样本 (可能都被排除)。")

	print(f"[2D-Test] 使用的数据根目录: {data_root}")
	print(f"[2D-Test] 测试样本总数: {len(test_samples)}, 可参与测试样本数: {num_test_samples}")
	print(f"[2D-Test] 类别总数: {num_classes}, 参与预测类别数: {len(active_indices)}, top_k={top_k}, 轮数={rounds}, 每首重复={repeats_per_sample}")
	if excluded_set:
		print(f"[2D-Test] 已排除风格: {sorted(excluded_set)}")
	else:
		print("[2D-Test] 已排除风格: 无")

	drum_encoder = AudioEncoder(n_mels=int(state["n_mels"]), embed_dim=int(state["embed_dim"])).to(device)
	timbre_encoder = AudioEncoder(n_mels=int(state["n_mels"]), embed_dim=int(state["embed_dim"])).to(device)
	drum_encoder.load_state_dict(state["drum_encoder_state_dict"])
	timbre_encoder.load_state_dict(state["timbre_encoder_state_dict"])
	drum_encoder.eval()
	timbre_encoder.eval()
	jazz_encoder = None
	if (
		"jazz_encoder_state_dict" in state
		and state.get("prototype_jazz_1d", None) is not None
		and 0 <= int(state.get("jazz_style_idx", -1)) < len(class_names)
	):
		jazz_encoder = AudioEncoder(n_mels=int(state["n_mels"]), embed_dim=int(state["embed_dim"])).to(device)
		jazz_encoder.load_state_dict(state["jazz_encoder_state_dict"])
		jazz_encoder.eval()

	for r in range(1, rounds + 1):
		print(f"\n[2D-Test] ==== 第 {r} 轮测试 ====")
		active_num_classes = len(active_indices)
		idx_to_active_pos = {orig_idx: pos for pos, orig_idx in enumerate(active_indices)}
		conf_mat = np.zeros((active_num_classes, active_num_classes), dtype=np.int64)
		per_class_total = np.zeros(active_num_classes, dtype=np.int64)
		per_class_correct = np.zeros(active_num_classes, dtype=np.int64)

		processed = 0
		for audio_path, label_idx in test_samples:
			true_idx = int(label_idx)
			if true_idx not in active_set:
				continue
			processed += 1
			true_pos = idx_to_active_pos[true_idx]
			per_class_total[true_pos] += 1
			try:
				avg_probs = _infer_probs_for_audio(
					audio_path=audio_path,
					state=state,
					drum_encoder=drum_encoder,
					timbre_encoder=timbre_encoder,
					jazz_encoder=jazz_encoder,
					device=device,
					repeats=repeats_per_sample,
				)
			except Exception as e:
				print(f"[2D-Test] 处理 {audio_path} 时出错: {e}")
				continue

			pred_score = avg_probs.copy()
			if excluded_indices:
				pred_score[list(excluded_indices)] = -np.inf
			pred_indices = [int(x) for x in np.argsort(-pred_score)[:top_k]]
			pred_top1 = int(pred_indices[0])
			pred_top1_pos = idx_to_active_pos[pred_top1]
			conf_mat[true_pos, pred_top1_pos] += 1

			is_correct = true_idx in pred_indices
			if not is_correct:
				true_name = class_names[true_idx]
				pred_names = [class_names[i] for i in pred_indices]
				if true_name == "Trap" and "Memphis" in pred_names:
					is_correct = True

			if is_correct:
				per_class_correct[true_pos] += 1

			progress = processed / num_test_samples
			bar_len = 30
			filled = int(bar_len * progress)
			bar = ">" * filled + "-" * (bar_len - filled)
			print(f"\r[2D-Test] 进度: |{bar}| {processed}/{num_test_samples} ({progress:6.2%})", end="", flush=True)

		print()
		total_correct = int(per_class_correct.sum())
		total_samples = int(per_class_total.sum())
		overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
		print("[2D-Test] 本轮整体 Top-{} 准确率: {:.2%}".format(top_k, overall_acc))
		print("[2D-Test] 各风格 Top-{} 准确率:".format(top_k))
		for pos, idx in enumerate(active_indices):
			name = class_names[idx]
			cnt = per_class_total[pos]
			acc = 0.0 if cnt == 0 else per_class_correct[pos] / cnt
			print(f"  [{idx:02d}] {name:20s}  acc={acc:.2%}  (correct {per_class_correct[pos]}/{cnt})")

		acc_list = []
		for pos, idx in enumerate(active_indices):
			name = class_names[idx]
			cnt = per_class_total[pos]
			acc = 0.0 if cnt == 0 else per_class_correct[pos] / cnt
			acc_list.append((acc, idx, name, int(per_class_correct[pos]), int(cnt)))
		acc_list.sort(key=lambda x: x[0], reverse=True)
		print("\n[2D-Test] 各风格准确率从高到低排序:")
		for acc, idx, name, correct_cnt, total_cnt in acc_list:
			print(f"  [{idx:02d}] {name:20s}  acc={acc:.2%}  (correct {correct_cnt}/{total_cnt})")

		print("\n[2D-Test] 本轮混淆矩阵 (行=真实, 列=预测, 仅包含参与风格):")
		header = "      " + " ".join([f"{i:4d}" for i in active_indices])
		print(header)
		for row_pos, i in enumerate(active_indices):
			row_vals = " ".join([f"{conf_mat[row_pos, col_pos]:4d}" for col_pos in range(active_num_classes)])
			print(f"[{i:02d}] {row_vals}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="2D 模型测试脚本 (使用训练时保留的测试集)")
	parser.add_argument("--data_root", type=str, default="", help="训练时使用的数据根目录, 为空则默认 Mymusic_all")
	parser.add_argument("--top_k", type=int, default=1, help="Top-K 准确率中的 K 值")
	parser.add_argument("--rounds", type=int, default=1, help="测试轮数")
	parser.add_argument("--repeats", type=int, default=10, help="单轮测试中, 每首歌重复测试的次数")
	parser.add_argument("--target_dir", type=str, default="", help="单个文件夹批量预测的目标文件夹路径")
	parser.add_argument("--output_txt", type=str, default="", help="批量预测结果输出 txt 路径, 为空则默认写入目标文件夹")
	parser.add_argument("--organize_errors", "--organize-errors", action="store_true", help="批量预测后将错分歌曲按 true_to_pred 规则归类到子文件夹")
	parser.add_argument("--organize_threshold", "--organize-threshold", type=float, default=0.75, help="错分风格归类覆盖率阈值 (0-1), 默认 0.75")
	parser.add_argument("--organize_root", "--organize-root", type=str, default="", help="错分归类输出根目录, 默认真实风格目录")
	parser.add_argument("--organize_mode", "--organize-mode", type=str, default="copy", help="错分归类方式: copy 或 move")
	parser.add_argument(
		"--exclude_styles",
		type=str,
		default="",
		help="排除风格, 用英文逗号分隔, 例如: Trap,Memphis",
	)
	return parser.parse_args()


def main() -> None:
	if len(sys.argv) == 1:
		print("[2D-Test] 进入交互模式, 将按提示设置参数 (直接回车使用默认值)")
		mode = _ask_text("请选择模式: 1-测试集评估, 2-单个文件夹批量预测 (默认 1)")
		if mode not in ("2", "folder", "batch", "dir"):
			mode = "1"

		data_root = _ask_text(f"请输入数据根目录 (默认: {DEFAULT_MUSIC_ROOT})") or DEFAULT_MUSIC_ROOT

		if mode == "2":
			target_dir = _ask_text("请输入要批量预测的文件夹路径").strip('"').strip("'")
			if not target_dir:
				print("未提供有效的文件夹路径, 结束批量预测模式。")
				return

			output_txt = _ask_text("输出txt路径(直接回车默认写到目标文件夹)").strip('"').strip("'")
			top_k = _ask_int("Top-K 中的 K (默认 1)", 1)
			repeats = _ask_int("每首歌重复次数 repeats (默认 10)", 10)
			excluded_styles = _split_excluded_styles(_ask_text("排除风格(英文逗号分隔, 直接回车表示不排除)"))

			print("\n[2D-Test] 将使用以下设置进行文件夹预测:")
			print(f"  data_root  = {data_root}")
			print(f"  target_dir = {target_dir}")
			print(f"  output_txt = {output_txt if output_txt else '默认写入目标文件夹'}")
			print(f"  top_k      = {top_k}")
			print(f"  repeats    = {repeats}")
			print(f"  exclude    = {excluded_styles if excluded_styles else '无'}")

			organize_raw = _ask_text("是否按主混淆风格创建错分子文件夹? (y/n, 默认 y)").strip().lower()
			organize_errors = organize_raw not in ("n", "no", "0", "false")
			organize_threshold = 0.75
			organize_root = ""
			organize_mode = "copy"
			if organize_errors:
				organize_threshold = _ask_float("错分归类覆盖率阈值 organize_threshold (0-1, 默认 0.75)", 0.75)
				organize_threshold = max(0.0, min(1.0, organize_threshold))
				organize_root = _ask_text("归类输出根目录(回车=默认真实风格目录)").strip('"').strip("'")
				mode_raw = _ask_text("归类方式 organize_mode (copy/move, 默认 copy)").strip().lower()
				if mode_raw in ("copy", "move"):
					organize_mode = mode_raw

			print(f"  organize_errors    = {organize_errors}")
			if organize_errors:
				print(f"  organize_threshold = {organize_threshold}")
				print(f"  organize_root      = {organize_root if organize_root else '默认真实风格目录'}")
				print(f"  organize_mode      = {organize_mode}")

			result_txt_path = predict_directory(
				target_dir=target_dir,
				data_root=data_root,
				output_txt=output_txt or None,
				top_k=top_k,
				repeats=repeats,
				excluded_styles=excluded_styles,
			)

			if not os.path.isfile(result_txt_path):
				raise FileNotFoundError(f"批量预测结果txt不存在: {result_txt_path}")

			from divide import run_divide_pipeline

			print("[2D-Test] 开始执行 divide 权重生成...")
			csv_path, summary_path, inferred_style, sample_count = run_divide_pipeline(
				result_txt=result_txt_path,
				organize_errors=organize_errors,
				organize_threshold=organize_threshold,
				organize_root=organize_root,
				organize_mode=organize_mode,
			)
			print(f"[2D-Test] divide 完成: style={inferred_style}, samples={sample_count}")
			print(f"[2D-Test] 权重CSV: {csv_path}")
			print(f"[2D-Test] 汇总TXT: {summary_path}")
			return

		top_k = _ask_int("Top-K 中的 K (默认 1)", 1)
		rounds = _ask_int("测试轮数 rounds (默认 1)", 1)
		repeats = _ask_int("每首歌重复测试次数 repeats (默认 10)", 10)
		excluded_styles = _split_excluded_styles(_ask_text("排除风格(英文逗号分隔, 直接回车表示不排除)"))

		print("\n[2D-Test] 将使用以下设置进行测试:")
		print(f"  data_root = {data_root}")
		print(f"  top_k    = {top_k}")
		print(f"  rounds   = {rounds}")
		print(f"  repeats  = {repeats}")
		print(f"  exclude  = {excluded_styles if excluded_styles else '无'}")

		run_test(
			data_root=data_root,
			top_k=top_k,
			rounds=rounds,
			repeats_per_sample=repeats,
			excluded_styles=excluded_styles,
		)
		return

	args = parse_args()
	data_root = args.data_root or DEFAULT_MUSIC_ROOT
	excluded_styles = _split_excluded_styles(args.exclude_styles)
	if args.target_dir.strip():
		result_txt_path = predict_directory(
			target_dir=args.target_dir.strip().strip('"').strip("'"),
			data_root=data_root,
			output_txt=args.output_txt.strip().strip('"').strip("'") or None,
			top_k=args.top_k,
			repeats=args.repeats,
			excluded_styles=excluded_styles,
		)
		if not os.path.isfile(result_txt_path):
			raise FileNotFoundError(f"批量预测结果txt不存在: {result_txt_path}")

		from divide import run_divide_pipeline

		print("[2D-Test] 开始执行 divide 权重生成...")
		csv_path, summary_path, inferred_style, sample_count = run_divide_pipeline(
			result_txt=result_txt_path,
			organize_errors=bool(args.organize_errors),
			organize_threshold=max(0.0, min(1.0, float(args.organize_threshold))),
			organize_root=args.organize_root.strip().strip('"').strip("'"),
			organize_mode=args.organize_mode,
		)
		print(f"[2D-Test] divide 完成: style={inferred_style}, samples={sample_count}")
		print(f"[2D-Test] 权重CSV: {csv_path}")
		print(f"[2D-Test] 汇总TXT: {summary_path}")
		return

	run_test(
		data_root=data_root,
		top_k=args.top_k,
		rounds=args.rounds,
		repeats_per_sample=args.repeats,
		excluded_styles=excluded_styles,
	)


if __name__ == "__main__":
	main()
